"""Kaggle GPU job: fine-tune two local fact extractors, then let each one take the exam.

QLoRA on the attached private dataset primnox-extractor-train (user messages from the
audited chat people, with 2 turns of context, -> the facts the audit kept; noisy copies
in train). Both T4s at once: Qwen 2.5 1.5B on GPU 0, Qwen 2.5 7B on GPU 1 (both
Apache-2.0). Each model then writes facts for LongMemEval's 467 knowledge-update
messages — never trained on — in the same {message: [facts]} shape as Gemini's
facts_lme_ku.json, for the head to score in a separate job.

Outputs in /kaggle/working: facts_lme_ku_<tag>.json, val_<tag>.json, adapters/<tag>/.
"""
import json
import os
import subprocess
import sys
import time

OUT = "/kaggle/working"
WORKER = r'''
import glob, json, os, sys, time, re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

tag, base, quant = sys.argv[1], sys.argv[2], sys.argv[3] == "4bit"
OUT = "/kaggle/working"
data_dir = os.path.dirname(glob.glob("/kaggle/input/**/train.jsonl", recursive=True)[0])
log = open(f"{OUT}/train_{tag}.log", "a", buffering=1)
def say(*a):
    print(*a, flush=True); print(*a, file=log)

tok = AutoTokenizer.from_pretrained(base)
tok.pad_token = tok.pad_token or tok.eos_token
kw = dict(dtype=torch.float16, device_map={"": 0})
if quant:
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(base, **kw)

def to_chat(ex):
    return {"prompt": [{"role": "user", "content": ex["prompt"]}],
            "completion": [{"role": "assistant", "content": ex["completion"]}]}
ds = load_dataset("json", data_files={"train": f"{data_dir}/train.jsonl", "val": f"{data_dir}/val.jsonl"})
ds = ds.map(to_chat)
import dataclasses
want = dict(output_dir=f"/tmp/run_{tag}", num_train_epochs=2, per_device_train_batch_size=4,
            gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=20,
            logging_steps=20, save_strategy="no", eval_strategy="epoch", fp16=True, bf16=False,
            max_length=1024, report_to=[], gradient_checkpointing=True)
known = {f.name for f in dataclasses.fields(SFTConfig)}
say(f"{tag}: config settings this library does not have, dropped: {sorted(set(want) - known) or 'none'}")
cfg = SFTConfig(**{k: v for k, v in want.items() if k in known})
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
t0 = time.time()
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"],
                     processing_class=tok, peft_config=lora)
# T4s train in fp16 with a gradient scaler, which cannot unscale bf16 gradients: every trainable
# (LoRA) weight goes to fp32; the frozen base stays fp16 / 4-bit.
cast = 0
for p in trainer.model.parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float(); cast += 1
say(f"{tag}: trainable tensors cast to fp32: {cast}; trainable params "
    f"{sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) / 1e6:.1f}M")
trainer.train()
say(f"{tag}: trained in {time.time() - t0:.0f}s; eval loss {trainer.evaluate().get('eval_loss')}")
trainer.model.save_pretrained(f"{OUT}/adapters/{tag}")
model = trainer.model.eval()

def extract(prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"\[.*\]", text, re.S)
    try:
        facts = json.loads(m.group(0)) if m else []
        return [str(f).strip() for f in facts if str(f).strip()], True
    except Exception:
        return [], False

# held-out people: how often it gets "no fact" right, and fact counts
val = [json.loads(l) for l in open(f"{data_dir}/val.jsonl", encoding="utf-8")]
agree_empty = bad = 0
counts = []
for ex in val:
    got, ok = extract(ex["prompt"]); bad += not ok
    gold = json.loads(ex["completion"])
    agree_empty += (not got) == (not gold)
    counts.append((len(gold), len(got)))
json.dump({"examples": len(val), "empty_agreement": agree_empty, "unparseable": bad, "counts": counts},
          open(f"{OUT}/val_{tag}.json", "w"))
say(f"{tag}: held-out people — fact/no-fact agreement {agree_empty}/{len(val)}, unparseable {bad}")

# the exam: LongMemEval knowledge-update messages, each alone (no earlier turns given)
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
sh("pip install -q -U peft trl bitsandbytes accelerate datasets")
# Kaggle preinstalls torchao 0.10; new peft refuses any torchao below 0.16 even when it is unused
sh("pip uninstall -q -y torchao || true")
sh("rm -rf /tmp/ms && git clone -q --depth 1 https://github.com/Primnox/Memory-system /tmp/ms")
sh("python -c \"import torch; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\"")
open("/tmp/worker.py", "w").write(WORKER)
jobs = [("q15", "Qwen/Qwen2.5-1.5B-Instruct", "fp16", "0"), ("q7", "Qwen/Qwen2.5-7B-Instruct", "4bit", "1")]
procs = []
for tag, base, quant, gpu in jobs:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    procs.append(subprocess.Popen([sys.executable, "/tmp/worker.py", tag, base, quant], env=env,
                                  stdout=open(f"{OUT}/worker_{tag}.out", "w"), stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]
for tag, *_ in jobs:
    sh(f"tail -n 25 {OUT}/worker_{tag}.out")
print(f"exit codes {codes}; done in {time.time() - t0:.0f}s", flush=True)
