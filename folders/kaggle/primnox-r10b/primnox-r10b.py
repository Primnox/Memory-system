"""Kaggle GPU job, round 10b: round 10a's question, answered cheaply and cleanly.

Round 10a (change rows given twice, ~60k items, 3 epochs) FROZE: its log stopped at epoch 3 step 2800
(34,204 s) and Kaggle killed it at 12 h. Its per-epoch checkpoint was written outside /kaggle/working,
so nothing survived. Fixes here:
  - the same weighting at half the cost: change rows once, filing rows HALVED (seeded 50%), so change:filing
    goes from round 9's 2.6:1 to 5.2:1, as round 10a intended, with ~30k items instead of ~60k;
  - two variants side by side, one per T4, so the effect of the weighting is not mixed with the new data:
      r10w  weighting only (round 9's people and rows, half the filing rows)
      r10f  r10w + who-is-who pilot rows + the who-is-who and two-person people as change scenarios
  - the model and its per-epoch checkpoint go straight to /kaggle/working;
  - a freeze alarm: no new training log line for 25 min -> stack dump (py-spy) + nvidia-smi, stop, and test
    the last epoch's checkpoint instead (reported as such);
  - fast tests first (blind v2 changes, LongMemEval, who-is-who, blind v2 filing), slow ones last (Dialogue
    NLI, REALTALK sweep).
Everything else is round 9's recipe (laya base, soft ratio 2, 3 epochs, micro-batch 8 x 4).
"""
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

OUT = "/kaggle/working"
VARIANTS = {
    "r10w": dict(gpu="0", scen=[], rows=[]),
    "r10f": dict(gpu="1", scen=["train_identity_fit.json", "train_duo.json"], rows=["fit_identity_rows.jsonl"]),
}

WORKER = r'''
import glob, json, os, shutil, signal, subprocess, sys, threading, time
tag, ms, rows = sys.argv[1], sys.argv[2], sys.argv[3:]
OUT = "/kaggle/working"
os.chdir(f"{ms}/system_one")
log = open(f"{OUT}/{tag}.log", "a", buffering=1)
def say(*a):
    print(*a, file=log, flush=True)
def sh(cmd, check=False):
    say(f"$ {cmd}")
    return subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/{tag}.log | tail -n 3"], check=check)
def one(pattern):
    found = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert found, pattern
    return found[0]

model = f"{OUT}/models/laya-memory-{tag}"
tlog = f"{OUT}/{tag}_train.log"
t0 = time.time()
p = subprocess.Popen(["bash", "-c", f"python train_laya.py --device cuda --rows {' '.join(rows)} --soft-ratio 2 --epochs 3 "
                      f"--micro-batch 8 --grad-accum 4 --output-dir {model} > {tlog} 2>&1"])
froze, last_size, last_change = False, -1, time.time()
deadline = float(os.environ.get("TRAIN_DEADLINE", "inf"))
while p.poll() is None:
    time.sleep(60)
    size = os.path.getsize(tlog) if os.path.exists(tlog) else 0
    if size != last_size:
        last_size, last_change = size, time.time()
    if time.time() > deadline:
        froze = True
        say(f"DEADLINE: training stopped at {time.time() - t0:.0f}s so the tests fit in Kaggle's 12 h")
        subprocess.run(["pkill", "-TERM", "-P", str(p.pid)]); p.terminate(); time.sleep(30); p.kill()
        break
    if time.time() - last_change > 25 * 60:
        froze = True
        say(f"FROZE: no training output for 25 min at {time.time() - t0:.0f}s")
        sh(f"py-spy dump --pid {p.pid} > {OUT}/{tag}_stack.txt 2>&1; for c in $(pgrep -P {p.pid}); do py-spy dump --pid $c >> {OUT}/{tag}_stack.txt 2>&1; done")
        sh(f"nvidia-smi >> {OUT}/{tag}_stack.txt 2>&1")
        subprocess.run(["pkill", "-TERM", "-P", str(p.pid)]); p.terminate(); time.sleep(30); p.kill()
        break
say(f"{tag} training {'FROZE' if froze else 'ended'} (exit {p.returncode}) after {time.time() - t0:.0f}s")
if froze or not os.path.exists(f"{model}/rl_agent_config.json"):
    if os.path.exists(f"{model}/checkpoint_latest/rl_agent_config.json"):
        model = f"{model}/checkpoint_latest"
        say(f"{tag}: testing the LAST EPOCH CHECKPOINT {model}")
    else:
        say(f"{tag}: no model to test"); sys.exit(1)
for leftover in ("train_items.pt",):
    f = os.path.join(f"{OUT}/models/laya-memory-{tag}", leftover)
    if os.path.exists(f): os.remove(f)

sh(f"python blind_pairs.py --model {model} --tag {tag}-clean --save-pairs")
open("run_llm_pieces.py", "w").write(open("/tmp/llm_runner.py").read())
sh(f"python run_llm_pieces.py {one('facts_lme_ku.json')} --model {model} --tag {tag}-lme-llm --sentences --files data/blind_external/pairs_lme_ku.jsonl")
sh(f"python /tmp/ideval.py {model} {tag}")
filer = open(one("r9_filer.py"), encoding="utf-8").read().replace("/tmp/ms", ms)
open(f"/tmp/filer_{tag}.py", "w").write(filer)
sh(f"python /tmp/filer_{tag}.py {model} {tag}")
sh(f"cd {OUT} && tar czf filed_{tag}.tgz filed_{tag} || true")
for f in glob.glob("results/blind_*" + tag + "*") + glob.glob("results/external_" + tag + "*"):
    shutil.copy(f, OUT)
if tag == "r10f":
    r9 = [q for q in glob.glob("/kaggle/input/**/laya-memory-r9", recursive=True) if os.path.isdir(q)]
    if r9:
        sh(f"python /tmp/ideval.py {r9[0]} r9")
sh(f"python external_eval.py --model {model} --tag {tag}-dnli --batch 64 --files "
   f"/tmp/dnli/pairs_dnli_test_full.jsonl /tmp/dnli/pairs_dnli_verified_test_full.jsonl")
for f in glob.glob("results/external_" + tag + "*"):
    shutil.copy(f, OUT)
sh(f"MS={ms} python /tmp/sweep.py {model} {tag}")
say(f"{tag} all done after {time.time() - t0:.0f}s")
'''


def sh(cmd: str, check: bool = True) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log | tail -n 5"], check=check)


def one(pattern: str) -> str:
    found = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert found, f"{pattern} not in the attached inputs"
    return found[0]


def patch(path: str, old: str, new: str) -> None:
    s = open(path, encoding="utf-8").read()
    assert old in s, f"{path}: {old[:60]}"
    open(path, "w", encoding="utf-8").write(s.replace(old, new))


def embedded(name: str) -> str:
    """IDEVAL / SWEEP / LLM_RUNNER are shipped in the dataset as the round-10 kernel's own code."""
    import ast
    src = open(one("primnox-r10.py"), encoding="utf-8").read()
    return next(n.value.value for n in ast.parse(src).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name)


def main() -> None:
    t0 = time.time()
    os.environ["PYTHONUNBUFFERED"] = "1"
    sh("pip install -q laya scikit-learn py-spy")
    sh("nvidia-smi -L")
    # the shared test scripts (sweep reads MS for its clone)
    open("/tmp/ideval.py", "w").write(embedded("IDEVAL"))
    open("/tmp/sweep.py", "w").write(embedded("SWEEP").replace('sys.path.insert(0, "/tmp/ms/system_one")',
                                                                 'sys.path.insert(0, os.environ["MS"] + "/system_one")'))
    open("/tmp/llm_runner.py", "w").write(embedded("LLM_RUNNER"))
    # Dialogue NLI pairs, built once (as in rounds 7-10a)
    sh("rm -rf /tmp/ms0 && git clone -q --depth 1 https://github.com/Primnox/Memory-system /tmp/ms0")
    sys.path.insert(0, "/tmp/ms0/system_one")
    from huggingface_hub import hf_hub_download
    from build_dnli import tidy
    os.makedirs("/tmp/dnli", exist_ok=True)
    for split in ("test", "verified_test"):
        path = hf_hub_download("xksteven/dialogue_nli", f"dialogue_nli/dialogue_nli/dialogue_nli_{split}.jsonl",
                               repo_type="dataset")
        text = open(path, encoding="utf-8").read()
        try:
            rows = json.loads(text)
        except ValueError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        with open(f"/tmp/dnli/pairs_dnli_{split}_full.jsonl", "w", encoding="utf-8") as f:
            for i, ex in enumerate(rows):
                old, new = tidy(ex["sentence2"]), tidy(ex["sentence1"])
                if len(old) < 8 or len(new) < 8:
                    continue
                f.write(json.dumps({"id": ex.get("id", f"{split}-{i}"), "source": f"dnli-{split}:{ex['label']}",
                                    "old_text": old, "new_text": new, "old_date": "2025-03-01",
                                    "new_date": "2025-06-01", "label": int(ex["label"] == "negative")},
                                   ensure_ascii=False) + "\n")
    # half the filing rows (seeded): change:filing 2.6:1 -> 5.2:1 without doubling the run
    filing = [l for l in open(one("train_filing_rows.jsonl"), encoding="utf-8") if l.strip()]
    half = random.Random(20261012).sample(filing, len(filing) // 2)
    open("/tmp/filing_half.jsonl", "w", encoding="utf-8").writelines(half)
    print(f"filing rows {len(filing)} -> {len(half)}", flush=True)
    open("/tmp/worker.py", "w").write(WORKER)
    import torch
    n_gpu = torch.cuda.device_count()
    print(f"GPUs: {n_gpu}", flush=True)
    variants = VARIANTS if n_gpu >= 2 else {"r10f": dict(VARIANTS["r10f"], gpu="0")}   # one GPU: the full variant only
    procs = {}
    for tag, v in variants.items():
        ms = f"/tmp/ms_{tag}"
        sh(f"rm -rf {ms} && cp -r /tmp/ms0 {ms}")
        tl = f"{ms}/system_one/train_laya.py"
        patch(tl, 'ours = sum(1 for r in hard if r["meta"].get("source") != "dnli")',
              'ours = sum(1 for r in hard if r["meta"].get("source") != "dnli" and r["meta"]["kind"] not in ("filing", "identity"))')
        data = f"{ms}/system_one/data"
        for f in glob.glob(f"{data}/train_*.json"):
            os.remove(f)
        for f in sorted(glob.glob("/kaggle/input/**/train_r7_*.json", recursive=True)) + [one(s) for s in v["scen"]]:
            shutil.copy(f, data)
        sh(f"cd {ms}/system_one && python build_train.py")
        rows = [f"{data}/train_rows.jsonl", "/tmp/filing_half.jsonl"] + [one(r) for r in v["rows"]]
        procs[tag] = subprocess.Popen([sys.executable, "/tmp/worker.py", tag, ms] + rows,
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=v["gpu"],
                                               TRAIN_DEADLINE=str(t0 + 9.5 * 3600)),
                                      stdout=open(f"{OUT}/{tag}.out", "w"), stderr=subprocess.STDOUT)
        print(f"{tag} started on GPU {v['gpu']}", flush=True)
    while any(p.poll() is None for p in procs.values()):
        time.sleep(300)
        for tag in procs:
            tl = f"{OUT}/{tag}_train.log"
            if os.path.exists(tl):
                last = [l for l in open(tl, encoding="utf-8", errors="replace").read().splitlines() if "step" in l][-1:]
                print(f"[{time.time() - t0:.0f}s] {tag}: {last[0][:80] if last else 'starting'}", flush=True)
    sh(f"cat {OUT}/r10w.log {OUT}/r10f.log | grep -vE 'it/s|%\\|' | tail -n 80", check=False)
    for tag in procs:
        ck = f"{OUT}/models/laya-memory-{tag}/checkpoint_latest"
        if os.path.isdir(ck) and os.path.exists(f"{OUT}/models/laya-memory-{tag}/rl_agent_config.json"):
            shutil.rmtree(ck)                                      # the final model is there; keep the output small
    print(f"exit codes { {t: p.returncode for t, p in procs.items()} }; done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
