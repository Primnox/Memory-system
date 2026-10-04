"""Kaggle GPU job: train the memory head and score it on the dev pairs.

Pushed with `kaggle kernels push -p system_one/kaggle`. Clones the public
repo (scripts + training rows), installs Laya, trains two variants with Laya's
own recipe, scores each on the validation pairs with screen.py, and leaves
results, logs and the fine-tuned checkpoints in /kaggle/working for
`kaggle kernels output`.
"""
import os
import shutil
import subprocess
import time

OUT = "/kaggle/working"
SRC = "/tmp/ms"
RUNS = [
    ("v2", "data/train_rows.jsonl data/dnli_rows.jsonl"),   # both writers + Dialogue NLI
    ("v2-nodnli", "data/train_rows.jsonl"),                 # same, without Dialogue NLI
]


def sh(cmd: str, log: str | None = None) -> None:
    print(f"$ {cmd}", flush=True)
    if log:
        cmd = f"set -o pipefail; ({cmd}) 2>&1 | tee -a {log}"
    subprocess.run(["bash", "-c", cmd], check=True)


t0 = time.time()
os.makedirs(f"{OUT}/logs", exist_ok=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sh("pip install -q laya scikit-learn", f"{OUT}/logs/setup.log")
sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}", f"{OUT}/logs/setup.log")
sh("python -c \"import torch, laya; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
   "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\"", f"{OUT}/logs/setup.log")
os.chdir(f"{SRC}/system_one")

for tag, rows in RUNS:
    log = f"{OUT}/logs/{tag}.log"
    model = f"models/laya-memory-{tag}"
    started = time.time()
    sh(f"python train_laya.py --device cuda --rows {rows} --soft-ratio 2 --epochs 3 "
       f"--micro-batch 8 --grad-accum 4 --output-dir {model}", log)
    print(f"{tag}: trained in {time.time() - started:.0f}s", flush=True)
    sh(f"python screen.py --scorers laya_ft --laya-ft {model} --tag {tag}", log)
    shutil.copy("results/screen_dev.json", f"{OUT}/screen_dev_{tag}.json")
    shutil.copy(f"results/scores_laya_ft_{tag}.json", f"{OUT}/scores_laya_ft_{tag}.json")
    for leftover in ("train_items.pt", "checkpoint_latest"):
        p = os.path.join(model, leftover)
        shutil.rmtree(p) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))
    shutil.copytree(model, f"{OUT}/models/laya-memory-{tag}")

print(f"all done in {time.time() - t0:.0f}s", flush=True)
