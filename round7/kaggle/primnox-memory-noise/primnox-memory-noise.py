"""Kaggle GPU job: does the memory head survive messy typing?

Blind v2 (never trained on) replayed through round 6 three times, identical labels,
only the typing differs: clean, noisy (typos, phone shorthand, fragments, fillers,
voice-to-text slips, code-switched tags) and heavy (the same at 1.5x strength).
The noisy copies come from the attached private dataset primnox-noisy-eval
(noisy.py; labels verified identical to the clean set before upload).
blind_pairs.py prints totals only.
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


def main() -> None:
    t0 = time.time()
    os.environ["PYTHONUNBUFFERED"] = "1"
    sh("pip install -q laya scikit-learn")
    sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}")
    os.chdir(f"{SRC}/system_one")
    weights = glob.glob("/kaggle/input/**/primnox-memory-head-r6/**/model.safetensors", recursive=True)
    assert weights, f"round 6 not found: {sorted(glob.glob('/kaggle/input/*/*/*'))[:20]}"
    model = os.path.dirname(weights[0])
    sets = {"clean": "../scripts/blind_memory/v2/test.json"}
    for name in ("v2_noisy", "v2_noisy_heavy"):
        found = glob.glob(f"/kaggle/input/**/{name}.json", recursive=True)
        assert found, f"{name}.json not in the attached datasets"
        sets[name.replace("v2_", "")] = found[0]
    for tag, data in sets.items():
        sh(f"python blind_pairs.py --data {data} --model {model} --tag r6-{tag}", check=False)
    for f in glob.glob("results/blind_*r6-*"):
        if os.path.isfile(f):
            shutil.copy(f, OUT)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


try:
    main()
except BaseException:
    import traceback
    with open(f"{OUT}/run.log", "a") as f:
        traceback.print_exc(file=f)
    traceback.print_exc()
    raise
