"""Kaggle GPU job: the round-4 memory head's retirements on LongMemEval end to end.

Inputs: the scenarios built by lme_e2e.py build (dataset primnox-lme-e2e, derived
from LongMemEval, MIT) and the round-4 checkpoint from the primnox-memory-head
kernel's output. Code: the public Memory-system repo. Writes the retired pairs
(ids only) for lme_e2e.py score.
"""
import glob
import os
import shutil
import subprocess
import time

OUT, SRC = "/kaggle/working", "/tmp/ms"


def sh(cmd: str, check: bool = True) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log"], check=check)


t0 = time.time()
os.environ["PYTHONUNBUFFERED"] = "1"
sh("pip install -q laya")
sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}")
model = glob.glob("/kaggle/input/**/laya-memory-r4", recursive=True)[0]
data = glob.glob("/kaggle/input/**/lme_e2e.json", recursive=True)[0]
os.makedirs("/tmp/lme_e2e", exist_ok=True)
shutil.copy(data, "/tmp/lme_e2e/lme_e2e.json")
os.chdir(f"{SRC}/system_one")
# no change labels in this data, so blind_pairs' recall line divides by zero after saving
sh(f"python blind_pairs.py --data /tmp/lme_e2e/lme_e2e.json --model {model} --tag r4 --save-pairs", check=False)
shutil.copy("results/blind_lme_e2e_r4.json", OUT)
print(f"done in {time.time() - t0:.0f}s", flush=True)
