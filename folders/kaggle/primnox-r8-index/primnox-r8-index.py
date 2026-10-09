"""Kaggle GPU job, round 8: a head that reads the index, and a local model that builds it.

GPU 0 — head r8i. Round 7's recipe (same people, same noisy copies, same settings), except each
statement carries where it is filed — "[about Sarah · lives in · also: Leeds]" — in one of five
statements dropped, so it also works without an index (primnox-memory-r8i-train). Scored on
blind v2 clean with no tags, with tags from a cloud clerk filing one fact at a time, and with tags from
the local clerk trained on GPU 1 (also one fact at a time); LongMemEval knowledge updates with
Gemini-extracted facts (no tags); Dialogue NLI (no tags). Round 7 is scored on the same tagged
file for reference.
GPU 1 — the clerk. Qwen 2.5 3B Instruct, QLoRA on the Gemini clerk's own filing calls of the
training people (primnox-clerk-train; never blind v2, LongMemEval or REALTALK). Checked against
Gemini on 8 held-out people, then it files blind v2 one fact at a time, as the app would.
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
CLERK_TAGS = f"{OUT}/blind_v2_filed_clerk.json"


def sh(cmd: str, check: bool = True, log: str = "run.log") -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/{log}"], check=check)


def one(pattern: str) -> str:
    found = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert found, f"{pattern} not in the attached inputs"
    return found[0]


LLM_RUNNER = '''
import json, sys
import external_eval
facts = json.load(open(sys.argv[1], encoding="utf-8"))
external_eval.statements = lambda text, keep_all_if_none=True: facts.get(text) or [text]
sys.argv = ["external_eval.py"] + sys.argv[2:]
external_eval.main()
'''


def patch(path: str, old: str, new: str) -> None:
    s = open(path, encoding="utf-8").read()
    assert old in s, f"{path}: {old[:60]}"
    open(path, "w", encoding="utf-8").write(s.replace(old, new))


def head() -> None:
    os.chdir(f"{SRC}/system_one")
    sys.path.insert(0, os.getcwd())
    # the index tag rides along in the state text; without one the text is exactly round 7's
    patch("screen.py",
          'def state_text(p: dict) -> str:\n'
          '    return (f\'Earlier ({p["old_date"]}) the user said: "{p["old_text"]}"\\n\'\n'
          '            f\'Later ({p["new_date"]}) the user said: "{p["new_text"]}"\')',
          'def state_text(p: dict) -> str:\n'
          '    old = f\' [{p["old_filed"]}]\' if p.get("old_filed") else ""\n'
          '    new = f\' [{p["new_filed"]}]\' if p.get("new_filed") else ""\n'
          '    return (f\'Earlier ({p["old_date"]}) the user said: "{p["old_text"]}"{old}\\n\'\n'
          '            f\'Later ({p["new_date"]}) the user said: "{p["new_text"]}"{new}\')')
    patch("build_train.py", '"old_date": old["date"], "new_date": new["date"]}',
          '"old_date": old["date"], "new_date": new["date"], "old_filed": old.get("filed"), "new_filed": new.get("filed")}')
    patch("blind_pairs.py", '"old_date": o["date"], "new_date": new["date"]}',
          '"old_date": o["date"], "new_date": new["date"], "old_filed": o.get("filed"), "new_filed": new.get("filed")}')

    for f in glob.glob("data/train_*.json"):
        os.remove(f)
    mix = sorted(glob.glob("/kaggle/input/**/train_r8i_*.json", recursive=True))
    assert mix, "round-8 training files are not attached"
    for f in mix:
        shutil.copy(f, "data/")
    sh("python build_train.py")
    sh("head -c 600 data/train_rows.jsonl; echo")
    r8 = "models/laya-memory-r8i"
    t0 = time.time()
    sh(f"python train_laya.py --device cuda --rows data/train_rows.jsonl --soft-ratio 2 --epochs 3 "
       f"--micro-batch 8 --grad-accum 4 --output-dir {r8}")
    print(f"r8i trained in {time.time() - t0:.0f}s", flush=True)
    for leftover in ("train_items.pt", "checkpoint_latest"):
        p = os.path.join(r8, leftover)
        shutil.rmtree(p) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))
    shutil.copytree(r8, f"{OUT}/{r8}")

    r7 = [p for p in glob.glob("/kaggle/input/**/laya-memory-r7", recursive=True) if os.path.isdir(p)][0]
    b1_tags = one("blind_v2_filed.json")
    sh(f"python blind_pairs.py --model {r8} --tag r8i-clean-notags", check=False)
    sh(f"python blind_pairs.py --data {b1_tags} --model {r8} --tag r8i-clean-b1tags", check=False)
    sh(f"python blind_pairs.py --data {b1_tags} --model {r7} --tag r7-clean-b1tags", check=False)
    open("run_llm_pieces.py", "w").write(LLM_RUNNER)
    sh(f"python run_llm_pieces.py {one('facts_lme_ku.json')} --model {r8} --tag r8i-lme-llm --sentences --files {LME}",
       check=False)
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
    sh(f"python external_eval.py --model {r8} --tag r8i-dnli --batch 64 --files "
       f"data/blind_external/pairs_dnli_test_full.jsonl data/blind_external/pairs_dnli_verified_test_full.jsonl",
       check=False)
    waited = 0
    while not os.path.exists(CLERK_TAGS) and waited < 4 * 3600:
        time.sleep(60)
        waited += 60
    if os.path.exists(CLERK_TAGS):
        sh(f"python blind_pairs.py --data {CLERK_TAGS} --model {r8} --tag r8i-clean-clerktags", check=False)
    for f in glob.glob("results/blind_*") + glob.glob("results/external_r8i*"):
        if os.path.isfile(f):
            shutil.copy(f, OUT)


CLERK = r'''
import glob, json, os, re, sys, time, dataclasses
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

OUT, BASE, MAXLEN = "/kaggle/working", "Qwen/Qwen2.5-3B-Instruct", 8192
src = os.path.dirname(glob.glob("/kaggle/input/**/held_out.json", recursive=True)[0])   # the clerk dataset
log = open(f"{OUT}/clerk.log", "a", buffering=1)
def say(*a):
    print(*a, flush=True); print(*a, file=log)

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={"": 0}, quantization_config=q4)

def to_chat(ex):
    return {"prompt": [{"role": "user", "content": ex["prompt"]}],
            "completion": [{"role": "assistant", "content": ex["completion"]}]}
ds = load_dataset("json", data_files={"train": f"{src}/train.jsonl", "val": f"{src}/val.jsonl"})
ds = ds.filter(lambda ex: len(tok(ex["prompt"] + ex["completion"])["input_ids"]) <= MAXLEN - 64)
say(f"clerk examples: train {len(ds['train'])}, val {len(ds['val'])} (longer than {MAXLEN} tokens dropped)")
ds = ds.map(to_chat, remove_columns=["person"])
want = dict(output_dir="/tmp/clerk", num_train_epochs=3, per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=10, logging_steps=10, save_strategy="no",
            eval_strategy="epoch", fp16=True, bf16=False, max_length=MAXLEN, report_to=[], gradient_checkpointing=True)
known = {f.name for f in dataclasses.fields(SFTConfig)}
cfg = SFTConfig(**{k: v for k, v in want.items() if k in known})
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"],
                     processing_class=tok, peft_config=lora)
for p in trainer.model.parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()
t0 = time.time()
trainer.train()
say(f"clerk trained in {time.time() - t0:.0f}s; eval loss {trainer.evaluate().get('eval_loss')}")
trainer.model.save_pretrained(f"{OUT}/adapters/clerk-q3")
model = trainer.model.eval()
tok.padding_side = "left"

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

def generate(prompts, sample=False):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    kw = dict(do_sample=True, temperature=0.5, top_p=0.95) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=1500, pad_token_id=tok.pad_token_id, **kw)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

# 1) agreement with Gemini on the held-out people's own filing calls
val = [json.loads(l) for l in open(f"{src}/val.jsonl", encoding="utf-8")]
n = same_slot = same_subject = parsed = 0
for i in range(0, len(val), 4):
    chunk = val[i:i + 4]
    for ex, text in zip(chunk, generate([ex["prompt"] for ex in chunk])):
        gold = json.loads(ex["completion"])["facts"]
        try:
            got = parse(text).get("facts", {}); parsed += 1
        except Exception:
            got = {}
        for fid, g in gold.items():
            n += 1
            e = got.get(fid) or {}
            s = str(e.get("slot", ""))
            same_slot += s.split(".")[-1] == g["slot"].split(".")[-1]
            same_subject += s.split(".")[0] == g["slot"].split(".")[0]
say(f"clerk vs Gemini on held-out people: {parsed}/{len(val)} replies parsed; per fact: same slot key "
    f"{same_slot}/{n} ({100 * same_slot / max(n, 1):.0f}%), same subject folder {same_subject}/{n} "
    f"({100 * same_subject / max(n, 1):.0f}%)")

# 2) blind v2 filed one fact at a time, as the app would
INSTRUCTION = val[0]["prompt"].split("\n\n## Folders so far")[0]
def prompt(state, batch):
    fs = "\n".join(json.dumps({"id": k, **v}, ensure_ascii=False) for k, v in state["folders"].items()) or "(none yet)"
    ss = "\n".join(f"{k} ({v})" for k, v in state["slots"].items()) or "(none yet)"
    facts = "\n".join(json.dumps({"id": f["id"], "date": f["date"], "text": f["text"]}, ensure_ascii=False) for f in batch)
    return f"{INSTRUCTION}\n\n## Folders so far\n{fs}\n\n## Slots so far\n{ss}\n\n## New facts\n{facts}"

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
    e = state["facts"].get(fid)
    if not e:
        return None
    who = lambda k: "the user" if state["folders"].get(k, {}).get("type") == "speaker" else state["folders"].get(k, {}).get("name", k)
    subject, key = e["slot"].split(".", 1) if "." in e["slot"] else (e["folders"][0], e["slot"])
    others = [who(k) for k in e["folders"] if k != subject and who(k) != "the user"][:2]
    return f"about {who(subject)} · {key.replace('_', ' ')}" + (f" · also: {', '.join(others)}" if others else "")

blind = json.load(open(glob.glob("/kaggle/input/**/blind_v2_filed.json", recursive=True)[0], encoding="utf-8"))
people = []
for sc in blind:
    who = sc["scenario"].split("-")[0].capitalize()
    people.append({"sc": sc, "k": 0, "tries": 0, "lost": 0,
                   "facts": sorted(sc["statements"], key=lambda f: f["date"]),
                   "state": {"folders": {"F1": {"name": who, "type": "speaker", "aliases": ["I", "me", "my", "the user"],
                                                "link": ""}}, "slots": {}, "facts": {}}})
t0 = time.time()
while True:
    ready = [p for p in people if p["k"] < len(p["facts"])]
    if not ready:
        break
    sample = any(p["tries"] for p in ready)
    for p, text in zip(ready, generate([prompt(p["state"], [p["facts"][p["k"]]]) for p in ready], sample)):
        batch = [p["facts"][p["k"]]]
        try:
            got = parse(text)
        except Exception:
            got = None
        if got is not None and ok(got, batch, p["state"]):
            merge(p["state"], got, batch); p["k"] += 1; p["tries"] = 0
        else:
            p["tries"] += 1
            if p["tries"] >= 3:
                p["state"]["slots"].setdefault("F1.unfiled", "many")
                p["state"]["facts"][batch[0]["id"]] = {"folders": ["F1"], "slot": "F1.unfiled"}
                p["lost"] += 1; p["k"] += 1; p["tries"] = 0
say(f"clerk filed blind v2 one fact at a time in {time.time() - t0:.0f}s; unfiled facts: {sum(p['lost'] for p in people)}")
os.makedirs(f"{OUT}/clerk_blind", exist_ok=True)
for p in people:
    name = re.sub(r"[^\w\-]", "_", p["sc"]["scenario"])
    os.makedirs(f"{OUT}/clerk_blind/{name}", exist_ok=True)
    json.dump(p["state"], open(f"{OUT}/clerk_blind/{name}/state.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for s in p["sc"]["statements"]:
        s["filed"] = tag(p["state"], s["id"])
json.dump(blind, open(f"{OUT}/blind_v2_filed_clerk.json.tmp", "w", encoding="utf-8"), ensure_ascii=False)
os.replace(f"{OUT}/blind_v2_filed_clerk.json.tmp", f"{OUT}/blind_v2_filed_clerk.json")
say("clerk done")
'''


def main() -> None:
    t0 = time.time()
    os.environ["PYTHONUNBUFFERED"] = "1"
    sh("pip install -q laya scikit-learn")
    sh("pip install -q -U peft trl bitsandbytes accelerate datasets")
    sh("pip uninstall -q -y torchao || true")
    sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}")
    open("/tmp/clerk.py", "w").write(CLERK)
    clerk = subprocess.Popen([sys.executable, "/tmp/clerk.py"], env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"),
                             stdout=open(f"{OUT}/clerk.out", "w"), stderr=subprocess.STDOUT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        head()
    finally:
        code = clerk.wait()
        sh(f"tail -n 30 {OUT}/clerk.out")
        print(f"clerk exit {code}; all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
