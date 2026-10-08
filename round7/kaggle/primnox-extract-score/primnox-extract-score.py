"""Kaggle GPU job: every extractor through the same LongMemEval exam, scored by heads r6 and r7.

LongMemEval knowledge-update pairs (421, 72 real updates), never trained on by anything
here. Each extractor's {message: [facts]} file replaces extract.py's splitter in
external_eval.py --sentences; nothing else changes. Extractors:
  gemini        Gemini 3.8 Flash, generic note-taker prompt         (primnox-lme-extract)
  ft-q15/ft-q7  Qwen 2.5 1.5B / 7B fine-tuned on the audited chats  (primnox-extract-exam output)
  pr-q3/pr-q7   Qwen 2.5 3B / 7B, same prompt, no training           (primnox-local-extract)
"""
import glob
import os
import shutil
import subprocess
import time

OUT, SRC = "/kaggle/working", "/tmp/ms"
LME = "data/blind_external/pairs_lme_ku.jsonl"
RUNNER = '''
import json, sys
import external_eval
facts = json.load(open(sys.argv[1], encoding="utf-8"))
external_eval.statements = lambda text, keep_all_if_none=True: facts.get(text) or [text]
sys.argv = ["external_eval.py"] + sys.argv[2:]
external_eval.main()
'''


def sh(cmd: str, check: bool = True) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log"], check=check)


def first(pattern: str, want_dir: bool = False) -> str | None:
    found = [p for p in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
             if os.path.isdir(p) == want_dir]
    return found[0] if found else None


t0 = time.time()
os.environ["PYTHONUNBUFFERED"] = "1"
sh("pip install -q laya scikit-learn")
sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}")
os.chdir(f"{SRC}/system_one")
open("run_pieces.py", "w").write(RUNNER)
r6 = first("primnox-memory-head-r6/**/model.safetensors")
models = {"r6": os.path.dirname(r6) if r6 else None, "r7": first("laya-memory-r7", want_dir=True)}
facts = {"gemini": first("facts_lme_ku.json"), "ft-q15": first("facts_lme_ku_q15.json"),
         "ft-q7": first("facts_lme_ku_q7.json"), "pr-q3": first("facts_lme_ku_pr-q3.json"),
         "pr-q7": first("facts_lme_ku_pr-q7.json")}
print(f"models: {models}\nfacts: {facts}", flush=True)
for mtag, model in models.items():
    if not model:
        continue
    for ftag, path in facts.items():
        if path:
            sh(f"python run_pieces.py {path} --model {model} --tag {mtag}-{ftag} --sentences --files {LME}",
               check=False)
for f in glob.glob("results/external_*"):
    if os.path.isfile(f):
        shutil.copy(f, OUT)
print(f"done in {time.time() - t0:.0f}s", flush=True)
