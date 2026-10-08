"""Kaggle GPU job: extractor v2 (1 and 2 passes) through the same LongMemEval exam, scored by head r7.

LongMemEval knowledge-update pairs (421, 72 real updates), never trained on by anything
here. Each extractor's {message: [facts]} file replaces extract.py's splitter in
external_eval.py --sentences; nothing else changes. Extractors (primnox-extractor-v2 output): Qwen 2.5 7B fine-tuned on the v2 data, 1 pass
(q7e1) and 2 passes (q7e2). Earlier results with the same head: Gemini facts 0.78, v1 7B 0.67.
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
models = {"r7": first("laya-memory-r7", want_dir=True)}
facts = {"ft2-q7e1": first("facts_lme_ku_q7e1.json"), "ft2-q7e2": first("facts_lme_ku_q7e2.json")}
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
