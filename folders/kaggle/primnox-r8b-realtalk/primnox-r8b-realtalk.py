"""Kaggle GPU job, round 8b: real chats (REALTALK) with the round-8 models.

GPU 1 — the local clerk trained in round 8 (Qwen 2.5 3B + adapter) files REALTALK's facts
(the Gemini-extracted facts of 10 real 21-day chats; private dataset primnox-folders-input,
facts only) FIVE facts per call in date order, never seeing later facts.
GPU 0 — the round-7 head (and round 8's, for comparison) run as the app runs it (15 most similar + 12 name-linked candidates),
without tags and with the clerk's tags; retirements are saved as (old, new, p) id pairs. Also on
blind v2 (no tags / tags from the cloud clerk's one-at-a-time filing / tags from round 8's local
clerk), so search can be scored with the round-8 head's own decisions.
REALTALK is a proof set: nothing here is trained or tuned on it. Its data never leaves Kaggle
except as fact ids, folder filings and scores.
"""
import glob
import json
import os
import subprocess
import sys
import time

OUT, SRC = "/kaggle/working", "/tmp/ms"
TAGS = f"{OUT}/realtalk_tags_clerk.json"

CLERK = r'''
import glob, json, os, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

OUT, BASE, STEP = "/kaggle/working", "Qwen/Qwen2.5-3B-Instruct", 5
adapter = [p for p in glob.glob("/kaggle/input/**/adapters/clerk-q3", recursive=True) if os.path.isdir(p)][0]
clerk_src = os.path.dirname(glob.glob("/kaggle/input/**/held_out.json", recursive=True)[0])
sets = json.load(open(glob.glob("/kaggle/input/**/sets.json", recursive=True)[0], encoding="utf-8"))
log = open(f"{OUT}/clerk.log", "a", buffering=1)
def say(*a):
    print(*a, flush=True); print(*a, file=log)

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "left"
q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={"": 0}, quantization_config=q4)
model = PeftModel.from_pretrained(model, adapter).eval()
val = [json.loads(l) for l in open(f"{clerk_src}/val.jsonl", encoding="utf-8")]
INSTRUCTION = val[0]["prompt"].split("\n\n## Folders so far")[0]

def parse(text):
    dec, found, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1; continue
        found.append(obj); i = end
    if not found:
        raise ValueError("no JSON")
    return found[-1]

def prompt(state, batch):
    fs = "\n".join(json.dumps({"id": k, **v}, ensure_ascii=False) for k, v in state["folders"].items()) or "(none yet)"
    ss = "\n".join(f"{k} ({v})" for k, v in state["slots"].items()) or "(none yet)"
    facts = "\n".join(json.dumps({"id": f["id"], "date": f["date"], "text": f["text"]}, ensure_ascii=False) for f in batch)
    return f"{INSTRUCTION}\n\n## Folders so far\n{fs}\n\n## Slots so far\n{ss}\n\n## New facts\n{facts}"

def generate(prompts, sample=False):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    kw = dict(do_sample=True, temperature=0.5, top_p=0.95) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=900, pad_token_id=tok.pad_token_id, **kw)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

def ok(got, batch, state):
    if not isinstance(got, dict) or not isinstance(got.get("facts"), dict):
        return False
    new = [f for f in got.get("new_folders", []) or [] if isinstance(f, dict)]
    if any(f.get("id") in state["folders"] or not f.get("id") for f in new):
        return False
    folders = set(state["folders"]) | {f.get("id") for f in new}
    for f in batch:
        e = got["facts"].get(f["id"])
        if not isinstance(e, dict) or not isinstance(e.get("folders"), list) or not e["folders"] \
                or any(x not in folders for x in e["folders"]) or not isinstance(e.get("slot"), str) \
                or e["slot"].split(".")[0] not in folders:
            return False
    return True

def merge(state, got, batch):
    for f in got.get("new_folders", []) or []:
        if isinstance(f, dict) and f.get("id"):
            state["folders"][f["id"]] = {"name": str(f.get("name", "")), "type": str(f.get("type", "")),
                                         "aliases": [a for a in (f.get("aliases") or []) if isinstance(a, str)],
                                         "link": str(f.get("link") or "")}
    for u in got.get("folder_updates", []) or []:
        cur = state["folders"].get(u.get("id")) if isinstance(u, dict) else None
        if cur:
            if u.get("name"):
                cur["aliases"] = sorted(set(cur["aliases"]) | {cur["name"]}); cur["name"] = str(u["name"])
            cur["aliases"] = sorted(set(cur["aliases"]) | {a for a in (u.get("add_aliases") or []) if isinstance(a, str)})
            if u.get("link"):
                cur["link"] = str(u["link"])
    for s in got.get("new_slots", []) or []:
        if isinstance(s, dict) and isinstance(s.get("key"), str):
            state["slots"].setdefault(s["key"], "one" if s.get("holds") == "one" else "many")
    for f in batch:
        e = got["facts"][f["id"]]
        state["facts"][f["id"]] = {"folders": list(e["folders"]), "slot": e["slot"]}
        state["slots"].setdefault(e["slot"], "many")

def tag(state, fid):
    # REALTALK has two named speakers: they are tagged by name, not as "the user"
    e = state["facts"].get(fid)
    if not e:
        return None
    who = lambda k: state["folders"].get(k, {}).get("name", k)
    subject, key = e["slot"].split(".", 1) if "." in e["slot"] else (e["folders"][0], e["slot"])
    others = [who(k) for k in e["folders"] if k != subject][:2]
    return f"about {who(subject)} · {key.replace('_', ' ')}" + (f" · also: {', '.join(others)}" if others else "")

people = []
for sc in sets["realtalk"]:
    facts = sorted(sc["statements"], key=lambda f: f["date"])
    people.append({"sc": sc, "k": 0, "tries": 0, "lost": 0, "batches": [facts[i:i + STEP] for i in range(0, len(facts), STEP)],
                   "state": {"folders": {}, "slots": {}, "facts": {}}})
t0, rounds = time.time(), 0
while True:
    ready = [p for p in people if p["k"] < len(p["batches"])]
    if not ready:
        break
    texts = generate([prompt(p["state"], p["batches"][p["k"]]) for p in ready], any(p["tries"] for p in ready))
    rounds += 1
    for p, text in zip(ready, texts):
        batch = p["batches"][p["k"]]
        try:
            got = parse(text)
        except Exception:
            got = None
        if got is not None and ok(got, batch, p["state"]):
            merge(p["state"], got, batch); p["k"] += 1; p["tries"] = 0
        else:
            p["tries"] += 1
            if p["tries"] >= 3:
                home = next(iter(p["state"]["folders"]), None)
                if home is None:
                    home = "F0"; p["state"]["folders"]["F0"] = {"name": "unfiled", "type": "thing", "aliases": [], "link": ""}
                p["state"]["slots"].setdefault(f"{home}.unfiled", "many")
                for f in batch:
                    p["state"]["facts"][f["id"]] = {"folders": [home], "slot": f"{home}.unfiled"}
                p["lost"] += len(batch); p["k"] += 1; p["tries"] = 0
    if rounds % 10 == 0:
        say(f"round {rounds}: {sum(len(p['batches']) - p['k'] for p in people)} batches left, {time.time() - t0:.0f}s")
say(f"clerk filed REALTALK in {time.time() - t0:.0f}s; unfiled facts: {sum(p['lost'] for p in people)} of "
    f"{sum(len(p['sc']['statements']) for p in people)}")
tags = {}
for p in people:
    name = re.sub(r"[^\w\-]", "_", p["sc"]["scenario"])
    os.makedirs(f"{OUT}/clerk_realtalk/{name}", exist_ok=True)
    json.dump(p["state"], open(f"{OUT}/clerk_realtalk/{name}/state.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tags[p["sc"]["scenario"]] = {f["id"]: tag(p["state"], f["id"]) for f in p["sc"]["statements"]}
json.dump(tags, open(f"{OUT}/realtalk_tags_clerk.json.tmp", "w", encoding="utf-8"), ensure_ascii=False)
os.replace(f"{OUT}/realtalk_tags_clerk.json.tmp", f"{OUT}/realtalk_tags_clerk.json")
say("clerk done")
'''

SWEEP = r'''
import glob, json, os, sys, time
sys.path.insert(0, "/tmp/ms/system_one")
from build_train import QUESTIONS, embed, entities, names_in
from laya.agent import Agent

OUT = "/kaggle/working"
sets = json.load(open(glob.glob("/kaggle/input/**/sets.json", recursive=True)[0], encoding="utf-8"))
people = sets["realtalk"]
MODELS = {k: [p for p in glob.glob(f"/kaggle/input/**/laya-memory-{k}", recursive=True) if os.path.isdir(p)][0]
          for k in ("r7", "r8i")}
agents = {}
def get_agent(k):
    if k not in agents:
        agents[k] = Agent(MODELS[k])
    return agents[k]
vec = embed(sorted({s["text"] for p in people + sets["blind"] for s in p["statements"]}))   # both sets swept
log = open(f"{OUT}/sweep.log", "a", buffering=1)

def state(o, n, tags):
    ot = f' [{tags[o["id"]]}]' if tags and tags.get(o["id"]) else ""
    nt = f' [{tags[n["id"]]}]' if tags and tags.get(n["id"]) else ""
    return (f'Earlier ({o["date"]}) the user said: "{o["text"]}"{ot}\n'
            f'Later ({n["date"]}) the user said: "{n["text"]}"{nt}')

def sweep(name, all_tags, people=people, set_name="realtalk"):
    agent = get_agent(name.split("-")[0])
    t0, out, scored = time.time(), {}, 0
    for p in people:
        tags = (all_tags or {}).get(p["scenario"])
        sts = sorted(p["statements"], key=lambda s: (s["date"], s["id"]))
        live, got = [], []
        for i, new in enumerate(sts):
            names = names_in(sts[:i + 1])
            ent_new = entities(new["text"], names)
            ranked = sorted(live, key=lambda o: -float(vec[o["text"]] @ vec[new["text"]]))
            cands = ranked[:15] + [o for o in ranked[15:] if entities(o["text"], names) & ent_new][:12]
            if cands:
                res = agent.predict_batch([state(o, new, tags) for o in cands], QUESTIONS)
                probs = [r["answers"]["relation"]["probabilities"]["replaces"] for r in res]
                retire = {o["id"]: pr for o, pr in zip(cands, probs) if pr >= 0.5}
                got += [[o, new["id"], round(pr, 4)] for o, pr in retire.items()]
                live = [o for o in live if o["id"] not in retire]
                scored += len(cands)
            live.append(new)
        out[p["scenario"]] = got
    json.dump(out, open(f"{OUT}/retired_{name}_{set_name}.json", "w"), indent=0)
    print(f"{name}: {sum(map(len, out.values()))} retired, {scored} pairs, {time.time() - t0:.0f}s", file=log, flush=True)

# round 8's head leaned on tags and lost elsewhere, so round 7 stays the head; this asks whether
# round 7 READING the local clerk's tags (who a fact is about, its slot) helps on real chats.
blind = sets["blind"]
def blind_tags(path):
    d = json.load(open(path, encoding="utf-8"))
    return {sc["scenario"]: {s["id"]: s.get("filed") for s in sc["statements"]} for sc in d}
b1 = blind_tags(glob.glob("/kaggle/input/**/blind_v2_filed.json", recursive=True)[0])
clerk_blind = blind_tags(glob.glob("/kaggle/input/**/blind_v2_filed_clerk.json", recursive=True)[0])
sweep("r7-b1tags", b1, blind, "blind")
sweep("r7-clerktags", clerk_blind, blind, "blind")
waited = 0
while not os.path.exists(f"{OUT}/realtalk_tags_clerk.json") and waited < 5 * 3600:
    time.sleep(60); waited += 60
if os.path.exists(f"{OUT}/realtalk_tags_clerk.json"):
    rt = json.load(open(f"{OUT}/realtalk_tags_clerk.json", encoding="utf-8"))
    sweep("r7-clerktags", rt)
    sweep("r8i-clerktags", rt)
'''


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log"], check=True)


t0 = time.time()
os.environ["PYTHONUNBUFFERED"] = "1"
sh("pip install -q laya scikit-learn")
sh("pip install -q -U peft bitsandbytes accelerate")
sh("pip uninstall -q -y torchao || true")
sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}")
open("/tmp/clerk.py", "w").write(CLERK)
open("/tmp/sweep.py", "w").write(SWEEP)
procs = [subprocess.Popen([sys.executable, "/tmp/clerk.py"], env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"),
                          stdout=open(f"{OUT}/clerk.out", "w"), stderr=subprocess.STDOUT),
         subprocess.Popen([sys.executable, "/tmp/sweep.py"], env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                          stdout=open(f"{OUT}/sweep.out", "w"), stderr=subprocess.STDOUT)]
codes = [p.wait() for p in procs]
sh(f"cat {OUT}/clerk.log {OUT}/sweep.log; tail -n 20 {OUT}/clerk.out {OUT}/sweep.out")
sh(f"cd {OUT} && tar czf clerk_realtalk.tgz clerk_realtalk")
print(f"exit codes {codes}; done in {time.time() - t0:.0f}s", flush=True)
