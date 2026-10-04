"""Kaggle GPU job, round 2: external blind tests + a second training seed.

1. Retrain the shipped variant (writers A + B, no Dialogue NLI) with the same
   recipe: a second run, so the blind result can be checked for seed variance.
2. Score untrained Laya and the retrained head on the external pairs
   (Dialogue NLI verified test, LongMemEval knowledge-update).
3. Replay blind v2 with the retrained head (seed-variance check of the first
   run's 95/112).
Results and logs land in /kaggle/working for `kaggle kernels output`.
"""
import os
import shutil
import subprocess
import time

OUT = "/kaggle/working"
SRC = "/tmp/ms"
MODEL = "models/laya-memory-seed2"


def sh(cmd: str, log: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {log}"], check=True)


t0 = time.time()
os.makedirs(f"{OUT}/logs", exist_ok=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sh("pip install -q laya scikit-learn", f"{OUT}/logs/setup.log")
sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}", f"{OUT}/logs/setup.log")
os.chdir(f"{SRC}/system_one")

sh(f"python train_laya.py --device cuda --rows data/train_rows.jsonl --soft-ratio 2 --epochs 3 "
   f"--micro-batch 8 --grad-accum 4 --output-dir {MODEL}", f"{OUT}/logs/train_seed2.log")
sh("python external_eval.py --model convaiinnovations/laya --tag zeroshot", f"{OUT}/logs/external.log")
sh(f"python external_eval.py --model {MODEL} --tag seed2", f"{OUT}/logs/external.log")
sh(f"python blind_pairs.py --model {MODEL} --tag seed2", f"{OUT}/logs/blind_v2_seed2.log")

for f in os.listdir("results"):
    if f.startswith(("external_", "blind_v2_seed2")):
        shutil.copy(os.path.join("results", f), OUT)
print(f"all done in {time.time() - t0:.0f}s", flush=True)
