"""Kaggle GPU job: the folder clerk run by an open model, Qwen 2.5 7B Instruct (4-bit), on both T4s.

Same filing rules (SPEC_FOLDERS.md) and same checks as the Gemini clerk (folders.py): facts go
in date order, 25 at a time, each call seeing only the folders and slots filed so far. The
attached private dataset primnox-folders-input holds facts only (no replacements, no
questions). Sets: chat (dev), blind v2 and REALTALK (proof), v3 long (change test).
People are split across the two GPUs; on each GPU up to BATCH people are filed at once.
A batch that fails its checks 3 times is filed as "unfiled" and counted.
Output: /kaggle/working/folders/<set>/<scenario>/state.json, same shape as folders.py.
"""
import json
import os
import subprocess
import sys
import time

OUT = "/kaggle/working"
WORKER = r'''
import glob, json, os, re, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

shard, shards = int(sys.argv[1]), int(sys.argv[2])
BASE, BATCH, STEP, TRIES = "Qwen/Qwen2.5-7B-Instruct", 6, 25, 3
OUT = "/kaggle/working"
src = os.path.dirname(glob.glob("/kaggle/input/**/sets.json", recursive=True)[0])
SPEC = open(f"{src}/SPEC_FOLDERS.md", encoding="utf-8").read()
SETS = json.load(open(f"{src}/sets.json", encoding="utf-8"))
log = open(f"{OUT}/worker_{shard}.log", "a", buffering=1)
def say(*a):
    print(*a, flush=True); print(*a, file=log)

people = [(s, p) for s, ps in SETS.items() for p in ps]
people.sort(key=lambda sp: -len(sp[1]["statements"]))
mine = people[shard::shards]

tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left"
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)).eval()

def parse(text):
    dec, found, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        found.append(obj); i = end
    if not found:
        raise ValueError("no JSON object")
    return found[-1]

def check(got, batch, state):
    if not isinstance(got, dict) or not isinstance(got.get("facts"), dict):
        return ["no facts object"]
    errs = []
    new = [f for f in got.get("new_folders", []) or [] if isinstance(f, dict)]
    clash = [f.get("id") for f in new if f.get("id") in state["folders"] or not f.get("id") or not f.get("name")]
    if clash:
        errs.append(f"new folder ids already used or missing: {clash[:3]}")
    folders = set(state["folders"]) | {f.get("id") for f in new}
    for f in batch:
        e = got["facts"].get(f["id"])
        if not isinstance(e, dict):
            errs.append(f"{f['id']} not filed"); continue
        fs, slot = e.get("folders"), e.get("slot")
        if not isinstance(fs, list) or not fs or any(x not in folders for x in fs):
            errs.append(f"{f['id']} folders {fs}")
        if not isinstance(slot, str) or slot.split(".")[0] not in folders:
            errs.append(f"{f['id']} slot {slot}")
    return errs

def merge(state, got, batch):
    for f in got.get("new_folders", []) or []:
        if isinstance(f, dict) and f.get("id"):
            state["folders"][f["id"]] = {"name": str(f.get("name", "")), "type": str(f.get("type", "")),
                                         "aliases": [str(a) for a in (f.get("aliases") or []) if isinstance(a, str)],
                                         "link": str(f.get("link") or "")}
    for u in got.get("folder_updates", []) or []:
        cur = state["folders"].get(u.get("id")) if isinstance(u, dict) else None
        if not cur:
            continue
        if u.get("name"):
            cur["aliases"] = sorted(set(cur["aliases"]) | {cur["name"]}); cur["name"] = str(u["name"])
        cur["aliases"] = sorted(set(cur["aliases"]) | {str(a) for a in (u.get("add_aliases") or []) if isinstance(a, str)})
        if u.get("link"):
            cur["link"] = str(u["link"])
    for s in got.get("new_slots", []) or []:
        if isinstance(s, dict) and isinstance(s.get("key"), str):
            state["slots"].setdefault(s["key"], "one" if s.get("holds") == "one" else "many")
    for f in batch:
        e = got["facts"][f["id"]]
        state["facts"][f["id"]] = {"folders": list(e["folders"]), "slot": e["slot"]}
        state["slots"].setdefault(e["slot"], "many")

def unfiled(state, batch):
    sp = [k for k, v in state["folders"].items() if v.get("type") == "speaker"]
    home = sp[0] if sp else "F0"
    state["folders"].setdefault(home, {"name": "unfiled", "type": "thing", "aliases": [], "link": ""})
    state["slots"].setdefault(f"{home}.unfiled", "many")
    for f in batch:
        state["facts"][f["id"]] = {"folders": [home], "slot": f"{home}.unfiled"}

def prompt(state, batch):
    folders = "\n".join(json.dumps({"id": k, **v}, ensure_ascii=False) for k, v in state["folders"].items()) or "(none yet)"
    slots = "\n".join(f"{k} ({v})" for k, v in state["slots"].items()) or "(none yet)"
    facts = "\n".join(json.dumps(f, ensure_ascii=False) for f in batch)
    return (f"{SPEC}\n## Folders so far\n{folders}\n\n## Slots so far\n{slots}\n\n## New facts\n{facts}\n\n"
            "Reply with the JSON object only, compact, no code fences.")

def generate(prompts, sample):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    kw = dict(do_sample=True, temperature=0.5, top_p=0.95) if sample else dict(do_sample=False)
    try:
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=2500, pad_token_id=tok.pad_token_id, **kw)
    except torch.cuda.OutOfMemoryError:
        del enc
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise
        say(f"  out of memory with {len(prompts)} prompts; halving")
        h = len(prompts) // 2
        return generate(prompts[:h], sample) + generate(prompts[h:], sample)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

class Person:
    def __init__(self, s, p):
        self.set, self.name = s, p["scenario"]
        self.state = p["first"]
        facts = sorted(p["statements"], key=lambda f: f["date"])
        self.batches = [facts[k:k + STEP] for k in range(0, len(facts), STEP)]
        self.k = self.tries = self.lost = 0
        self.dir = os.path.join(OUT, "folders", s, re.sub(r"[^\w\-]", "_", self.name))
        os.makedirs(self.dir, exist_ok=True)
    def save(self):
        self.state["unfiled_facts"] = self.lost
        json.dump(self.state, open(os.path.join(self.dir, "state.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

ps = [Person(s, p) for s, p in mine]
say(f"shard {shard}: {len(ps)} people, {sum(len(p.batches) for p in ps)} batches")
t0, rounds, fails = time.time(), 0, 0
while True:
    ready = [p for p in ps if p.k < len(p.batches)]
    if not ready:
        break
    # retries first (a greedy retry would repeat itself, so a group with one samples), then the
    # longest chains, which finish last
    ready.sort(key=lambda p: (p.tries == 0, -(len(p.batches) - p.k)))
    group = ready[:BATCH]
    group.sort(key=lambda p: len(prompt(p.state, p.batches[p.k])))
    texts = generate([prompt(p.state, p.batches[p.k]) for p in group], sample=any(p.tries > 0 for p in group))
    rounds += 1
    for p, text in zip(group, texts):
        batch = p.batches[p.k]
        try:
            got = parse(text)
            errs = check(got, batch, p.state)
            if errs:
                raise ValueError(f"{len(errs)} problems: {errs[:3]}")
            merge(p.state, got, batch)
            if p.k == 0:
                open(os.path.join(p.dir, "b0_raw.txt"), "w", encoding="utf-8").write(text)
            p.k += 1; p.tries = 0
        except Exception as e:
            p.tries += 1; fails += 1
            if fails <= 40:
                say(f"  {p.set}/{p.name} b{p.k}: try {p.tries} failed: {str(e)[:200]} | {text[:200]!r}")
            if p.tries >= TRIES:
                unfiled(p.state, batch); p.lost += len(batch); p.k += 1; p.tries = 0
        p.save()
    left = sum(len(p.batches) - p.k for p in ps)
    say(f"shard {shard} round {rounds}: {len(group)} filed or retried, {left} batches left, {time.time() - t0:.0f}s")
for p in ps:
    say(f"{p.set}/{p.name}: {len(p.state['folders'])} folders, {len(p.state['slots'])} slots, "
        f"{len(p.state['facts'])} facts, {p.lost} unfiled")
say(f"shard {shard} done in {time.time() - t0:.0f}s")
'''


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log"], check=True)


t0 = time.time()
os.environ["PYTHONUNBUFFERED"] = "1"
sh("pip install -q -U transformers bitsandbytes accelerate")
sh("python -c \"import torch; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\"")
open("/tmp/worker.py", "w").write(WORKER)
n = 2
procs = [subprocess.Popen([sys.executable, "/tmp/worker.py", str(i), str(n)],
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)),
                          stdout=open(f"{OUT}/worker_{i}.out", "w"), stderr=subprocess.STDOUT) for i in range(n)]
codes = [p.wait() for p in procs]
for i in range(n):
    sh(f"tail -n 40 {OUT}/worker_{i}.out")
sh(f"cd {OUT} && tar czf folders.tgz folders")
print(f"exit codes {codes}; done in {time.time() - t0:.0f}s", flush=True)
