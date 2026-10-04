"""Kaggle GPU job: retrain the memory head and run every evaluation.

Set TAG / ROWS below for each round (round 2: seed2 on writers A+B; round 3:
v3 on writers A+B+C, which adds long chatty messages and restatements).

1. Retrain with Laya's recipe on ROWS (no Dialogue NLI).
2. Score it on the external pairs (Dialogue NLI verified test, LongMemEval
   knowledge-update); untrained Laya too when ZERO_SHOT.
3. Replay blind v2 with it, and score it on the dev shortlist.
Results and logs land in /kaggle/working for `kaggle kernels output`.
"""
import os
import shutil
import subprocess
import time

OUT = "/kaggle/working"
SRC = "/tmp/ms"
TAG = "r4"   # round 4: writers A+B+C, plus fact extraction for long messages
ROWS = "data/train_rows.jsonl"
MODEL = f"models/laya-memory-{TAG}"
ZERO_SHOT = False   # already scored in round 2


def sh(cmd: str, log: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {log}"], check=True)


t0 = time.time()
os.makedirs(f"{OUT}/logs", exist_ok=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sh("pip install -q laya scikit-learn", f"{OUT}/logs/setup.log")
sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}", f"{OUT}/logs/setup.log")
os.chdir(f"{SRC}/system_one")

sh(f"python train_laya.py --device cuda --rows {ROWS} --soft-ratio 2 --epochs 3 "
   f"--micro-batch 8 --grad-accum 4 --output-dir {MODEL}", f"{OUT}/logs/train_{TAG}.log")
if ZERO_SHOT:
    sh("python external_eval.py --model convaiinnovations/laya --tag zeroshot", f"{OUT}/logs/external.log")
sh(f"python external_eval.py --model {MODEL} --tag {TAG}", f"{OUT}/logs/external.log")
sh(f"python external_eval.py --model {MODEL} --tag {TAG} --sentences", f"{OUT}/logs/external.log")
sh(f"python blind_pairs.py --model {MODEL} --tag {TAG}", f"{OUT}/logs/blind_v2_{TAG}.log")
sh(f"python screen.py --scorers laya_ft --laya-ft {MODEL} --tag {TAG}", f"{OUT}/logs/dev_{TAG}.log")

for f in os.listdir("results"):
    if f.startswith(("external_", f"blind_v2_{TAG}", "screen_dev")):
        shutil.copy(os.path.join("results", f), OUT)
for leftover in ("train_items.pt", "checkpoint_latest"):
    path = os.path.join(MODEL, leftover)
    shutil.rmtree(path) if os.path.isdir(path) else (os.path.exists(path) and os.remove(path))
shutil.copytree(MODEL, f"{OUT}/{MODEL}")
print(f"all done in {time.time() - t0:.0f}s", flush=True)
