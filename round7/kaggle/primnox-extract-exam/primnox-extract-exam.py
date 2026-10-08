"""Kaggle GPU job: the fine-tuned extractors take the exam (no training here).

Loads the LoRA adapters the primnox-extractor job saved (q15 = Qwen 2.5 1.5B, q7 = Qwen 2.5 7B),
one model per T4 in parallel, and writes facts for:
  - the 4 held-out chat people (fact / no-fact agreement with the audited answer key)
  - LongMemEval's 467 knowledge-update messages, never trained on, in the same
    {message: [facts]} shape as Gemini's facts_lme_ku.json, for the head to score.
"""
import os
import subprocess
import sys
import time

OUT = "/kaggle/working"
WORKER = r'''
import glob, json, os, sys, time, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

tag, base, quant = sys.argv[1], sys.argv[2], sys.argv[3] == "4bit"
OUT = "/kaggle/working"
log = open(f"{OUT}/exam_{tag}.log", "a", buffering=1)
def say(*a):
    print(*a, flush=True); print(*a, file=log)

adapter = [p for p in glob.glob(f"/kaggle/input/**/adapters/{tag}", recursive=True) if os.path.isdir(p)][0]
data_dir = os.path.dirname(glob.glob("/kaggle/input/**/val.jsonl", recursive=True)[0])
tok = AutoTokenizer.from_pretrained(base)
tok.pad_token = tok.pad_token or tok.eos_token
kw = dict(dtype=torch.float16, device_map={"": 0})
if quant:
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, **kw), adapter).eval()
say(f"{tag}: loaded {base} + adapter {adapter}")

def extract(prompt):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"\[.*\]", text, re.S)
    try:
        facts = json.loads(m.group(0)) if m else []
        return [str(f).strip() for f in facts if str(f).strip()], True
    except Exception:
        return [], False

val = [json.loads(l) for l in open(f"{data_dir}/val.jsonl", encoding="utf-8")]
agree = bad = 0
for ex in val:
    got, ok = extract(ex["prompt"]); bad += not ok
    agree += (not got) == (not json.loads(ex["completion"]))
say(f"{tag}: held-out people — fact/no-fact agreement {agree}/{len(val)}, unparseable {bad}")

src = "/tmp/ms/system_one/data/blind_external/pairs_lme_ku.jsonl"
INSTR = val[0]["prompt"].split("\n\nDate:")[0]
msgs = {}
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    for side in ("old", "new"):
        msgs.setdefault(r[f"{side}_text"], r[f"{side}_date"])
facts, t0 = {}, time.time()
for i, (text, date) in enumerate(msgs.items()):
    p = f"{INSTR}\n\nDate: {date}\nEarlier turns:\n(start of the chat)\n\nUser's latest message:\n{text}\n\nFacts:"
    facts[text], _ = extract(p)
    if i % 50 == 0:
        say(f"{tag}: exam {i}/{len(msgs)} ({time.time() - t0:.0f}s)")
json.dump(facts, open(f"{OUT}/facts_lme_ku_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n = [len(v) for v in facts.values()]
say(f"{tag}: exam done — {len(facts)} messages, {sum(n)} facts, {sum(1 for x in n if x == 0)} with none")
'''


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log"], check=True)


t0 = time.time()
os.environ["PYTHONUNBUFFERED"] = "1"
sh("pip install -q -U peft bitsandbytes accelerate")
sh("pip uninstall -q -y torchao || true")
sh("rm -rf /tmp/ms && git clone -q --depth 1 https://github.com/Primnox/Memory-system /tmp/ms")
open("/tmp/worker.py", "w").write(WORKER)
jobs = [("q15", "Qwen/Qwen2.5-1.5B-Instruct", "fp16", "0"), ("q7", "Qwen/Qwen2.5-7B-Instruct", "4bit", "1")]
procs = [subprocess.Popen([sys.executable, "/tmp/worker.py", tag, base, quant],
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                          stdout=open(f"{OUT}/worker_{tag}.out", "w"), stderr=subprocess.STDOUT)
         for tag, base, quant, gpu in jobs]
codes = [p.wait() for p in procs]
for tag, *_ in jobs:
    sh(f"grep -v 'it/s' {OUT}/worker_{tag}.out | tail -n 12")
print(f"exit codes {codes}; done in {time.time() - t0:.0f}s", flush=True)
