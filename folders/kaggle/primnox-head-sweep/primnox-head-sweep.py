"""Kaggle GPU job: the round-7 memory head run as the app runs it, with more candidates.

Facts arrive in date order. Each new fact is put to the head against candidates from the
LIVE (not yet retired) facts; P(replaces) >= 0.5 retires a candidate. Policies:
  app            today's app: the 15 most similar + up to 12 sharing a named person or thing
                 (memory/head.py candidates(); blind_pairs.py)
  slot-<clerk>   app, plus up to 20 live facts filed in the SAME SLOT as the new fact by that
                 clerk's filing (most similar first): "run the head more often"
Labels are not used for any decision; the outputs are the retired (old, new, p) id pairs per
person, scored locally. Input (private dataset primnox-head-sweep-input): sets.json (facts
only) and slots.json ({clerk: {set: {scenario: {fact id: slot}}}}).
"""
import json
import os
import subprocess
import sys
import time

OUT = "/kaggle/working"
WORKER = r'''
import glob, json, os, sys, time
sys.path.insert(0, "/tmp/ms/system_one")
from build_train import QUESTIONS, embed, entities, names_in
from screen import state_text
from laya.agent import Agent

jobs = json.loads(sys.argv[1])
OUT = "/kaggle/working"
src = os.path.dirname(glob.glob("/kaggle/input/**/sets.json", recursive=True)[0])
SETS = json.load(open(f"{src}/sets.json", encoding="utf-8"))
SLOTS = json.load(open(f"{src}/slots.json", encoding="utf-8"))
model = [p for p in glob.glob("/kaggle/input/**/laya-memory-r7", recursive=True) if os.path.isdir(p)][0]
agent = Agent(model)
K, LINKED, SAME_SLOT, THRESHOLD = 15, 12, 20, 0.5
log = open(f"{OUT}/sweep.log", "a", buffering=1)

for policy, set_name in jobs:
    clerk = policy.split("-", 1)[1] if policy.startswith("slot-") else None
    people = SETS[set_name]
    vec = embed(sorted({s["text"] for p in people for s in p["statements"]}))
    result, t0, scored = {}, time.time(), 0
    for p in people:
        slot = (SLOTS.get(clerk, {}).get(set_name, {}).get(p["scenario"]) if clerk else None)
        if clerk and not slot:
            continue
        sts = sorted(p["statements"], key=lambda s: (s["date"], s["id"]))
        live, got = [], []
        for i, new in enumerate(sts):
            names = names_in(sts[:i + 1])
            ent_new = entities(new["text"], names)
            ranked = sorted(live, key=lambda o: -float(vec[o["text"]] @ vec[new["text"]]))
            cands = ranked[:K]
            cands += [o for o in ranked[K:] if entities(o["text"], names) & ent_new][:LINKED]
            if slot:
                have = {o["id"] for o in cands}
                cands += [o for o in ranked if o["id"] not in have and slot.get(o["id"]) == slot.get(new["id"])][:SAME_SLOT]
            if cands:
                states = [state_text({"old_text": o["text"], "new_text": new["text"],
                                      "old_date": o["date"], "new_date": new["date"]}) for o in cands]
                res = agent.predict_batch(states, QUESTIONS)
                probs = [r["answers"]["relation"]["probabilities"]["replaces"] for r in res]
                retire = {o["id"]: pr for o, pr in zip(cands, probs) if pr >= THRESHOLD}
                got += [[o, new["id"], round(pr, 4)] for o, pr in retire.items()]
                live = [o for o in live if o["id"] not in retire]
                scored += len(cands)
            live.append(new)
        result[p["scenario"]] = got
    json.dump(result, open(f"{OUT}/retired_{policy}_{set_name}.json", "w", encoding="utf-8"), indent=0)
    print(f"{policy} {set_name}: {len(result)} people, {sum(map(len, result.values()))} retired, "
          f"{scored} pairs scored in {time.time() - t0:.0f}s", file=log, flush=True)
'''


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log"], check=True)


t0 = time.time()
os.environ["PYTHONUNBUFFERED"] = "1"
sh("pip install -q laya scikit-learn")
sh("rm -rf /tmp/ms && git clone -q --depth 1 https://github.com/Primnox/Memory-system /tmp/ms")
open("/tmp/worker.py", "w").write(WORKER)
SETS = ["blind", "chat", "v3long", "realtalk"]
# run 1: today's head everywhere, and same-slot candidates from Gemini's filing (blind, chat);
# run 2 (after the Qwen filing): same-slot candidates from Qwen's filing
PLAN = os.environ.get("PLAN") or "1"
plan = ([[("app", "blind"), ("app", "chat"), ("app", "v3long")],
         [("slot-gemini", "blind"), ("slot-gemini", "chat"), ("app", "realtalk")]] if PLAN == "1" else
        [[("slot-qwen", "blind"), ("slot-qwen", "chat"), ("slot-qwen", "v3long")], [("slot-qwen", "realtalk")]])
procs = [subprocess.Popen([sys.executable, "/tmp/worker.py", json.dumps(jobs)],
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)),
                          stdout=open(f"{OUT}/worker_{i}.out", "w"), stderr=subprocess.STDOUT)
         for i, jobs in enumerate(plan)]
codes = [p.wait() for p in procs]
sh(f"cat {OUT}/sweep.log; tail -n 15 {OUT}/worker_0.out {OUT}/worker_1.out")
print(f"exit codes {codes}; done in {time.time() - t0:.0f}s", flush=True)
