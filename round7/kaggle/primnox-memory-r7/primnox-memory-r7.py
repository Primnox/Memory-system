"""Kaggle GPU job: round 7 of the memory head — every dataset at once — and every proof.

Train: Laya's recipe on the private dataset primnox-memory-r7-train (existing people,
v3, audited chat facts and mixed-language people, plus noisy copies of all of them;
disputed pairs skipped). The repo's own train_*.json are removed first: the mix
already holds them.

Proofs, on round 6 and round 7 side by side, none of them trained on:
  blind v2 replay      clean, noisy and heavy-noise copies (primnox-noisy-eval)
  LongMemEval KU       whole messages, rule-split, and Gemini-extracted facts (primnox-lme-extract)
  Dialogue NLI         full test and verified test splits
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

OUT, SRC = "/kaggle/working", "/tmp/ms"
LME = "data/blind_external/pairs_lme_ku.jsonl"


def sh(cmd: str, check: bool = True) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log"], check=check)


def one(pattern: str) -> str:
    found = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert found, f"{pattern} not in the attached inputs: {sorted(glob.glob('/kaggle/input/*/*/*'))[:20]}"
    return found[0]


LLM_RUNNER = '''
import json, sys
import external_eval
facts = json.load(open(sys.argv[1], encoding="utf-8"))
external_eval.statements = lambda text, keep_all_if_none=True: facts.get(text) or [text]
sys.argv = ["external_eval.py"] + sys.argv[2:]
external_eval.main()
'''


def main() -> None:
    t0 = time.time()
    os.environ["PYTHONUNBUFFERED"] = "1"
    sh("pip install -q laya scikit-learn")
    sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}")
    os.chdir(f"{SRC}/system_one")
    sys.path.insert(0, os.getcwd())

    # training rows from the mix
    for f in glob.glob("data/train_*.json"):
        os.remove(f)
    mix = sorted(glob.glob("/kaggle/input/**/train_r7_*.json", recursive=True))
    assert mix, "the round-7 training files are not attached"
    for f in mix:
        shutil.copy(f, "data/")
    print(f"training files: {[os.path.basename(f) for f in mix]}", flush=True)
    sh("python build_train.py")
    r7 = "models/laya-memory-r7"
    started = time.time()
    sh(f"python train_laya.py --device cuda --rows data/train_rows.jsonl --soft-ratio 2 --epochs 3 "
       f"--micro-batch 8 --grad-accum 4 --output-dir {r7}")
    print(f"r7 trained in {time.time() - started:.0f}s", flush=True)
    for leftover in ("train_items.pt", "checkpoint_latest"):
        p = os.path.join(r7, leftover)
        shutil.rmtree(p) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))
    shutil.copytree(r7, f"{OUT}/{r7}")

    # Dialogue NLI test splits, as the bench job builds them
    from huggingface_hub import hf_hub_download
    from build_dnli import tidy
    for split in ("test", "verified_test"):
        path = hf_hub_download("xksteven/dialogue_nli", f"dialogue_nli/dialogue_nli/dialogue_nli_{split}.jsonl",
                               repo_type="dataset")
        text = open(path, encoding="utf-8").read()
        try:
            rows = json.loads(text)
        except ValueError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        with open(f"data/blind_external/pairs_dnli_{split}_full.jsonl", "w", encoding="utf-8") as f:
            for i, ex in enumerate(rows):
                old, new = tidy(ex["sentence2"]), tidy(ex["sentence1"])
                if len(old) < 8 or len(new) < 8:
                    continue
                f.write(json.dumps({"id": ex.get("id", f"{split}-{i}"), "source": f"dnli-{split}:{ex['label']}",
                                    "old_text": old, "new_text": new, "old_date": "2025-03-01",
                                    "new_date": "2025-06-01", "label": int(ex["label"] == "negative")},
                                   ensure_ascii=False) + "\n")
    with open("run_llm_pieces.py", "w") as f:
        f.write(LLM_RUNNER)

    models = {"r6": os.path.dirname(one("primnox-memory-head-r6/**/model.safetensors")), "r7": r7}
    noisy, heavy, facts = one("v2_noisy.json"), one("v2_noisy_heavy.json"), one("facts_lme_ku.json")
    dnli = ("data/blind_external/pairs_dnli_test_full.jsonl data/blind_external/pairs_dnli_verified_test_full.jsonl")
    for tag, model in models.items():
        sh(f"python blind_pairs.py --model {model} --tag {tag}-clean", check=False)
        sh(f"python blind_pairs.py --data {noisy} --model {model} --tag {tag}-noisy", check=False)
        sh(f"python blind_pairs.py --data {heavy} --model {model} --tag {tag}-heavy", check=False)
        sh(f"python external_eval.py --model {model} --tag {tag}-lme-whole --files {LME}", check=False)
        sh(f"python external_eval.py --model {model} --tag {tag}-lme-rules --sentences --files {LME}", check=False)
        sh(f"python run_llm_pieces.py {facts} --model {model} --tag {tag}-lme-llm --sentences --files {LME}",
           check=False)
        sh(f"python external_eval.py --model {model} --tag {tag}-dnli --batch 64 --files {dnli}", check=False)
    for f in glob.glob("results/*"):
        if os.path.isfile(f) and ("r6-" in f or "r7-" in f):
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
