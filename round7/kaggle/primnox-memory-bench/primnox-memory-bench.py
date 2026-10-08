"""Kaggle GPU job: the memory head's benchmarks at full size, round 4 against round 6.

Everything here is data no round trained on:
  Dialogue NLI   the whole test split (16,500 crowdworker pairs, MIT) and the
                 verified test split; contradiction -> should replace
  LongMemEval    knowledge-update pairs (421, MIT), as whole messages and split
                 into statements; and the end-to-end replay (136 questions),
                 whose retirements are scored through the app on the PC
                 (system_one/lme_e2e.py score)
Code: the public Memory-system repo. Models: round 4 from the primnox-memory-head
kernel's output, round 6 from the primnox-memory-head-r6 dataset.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
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
    sys.path.insert(0, os.getcwd())

    # Dialogue NLI test splits, straight from the public dataset
    from huggingface_hub import hf_hub_download  # noqa: E402
    from build_dnli import tidy  # noqa: E402
    for split in ("test", "verified_test"):
        path = hf_hub_download("xksteven/dialogue_nli", f"dialogue_nli/dialogue_nli/dialogue_nli_{split}.jsonl",
                               repo_type="dataset")
        text = open(path, encoding="utf-8").read()
        try:
            rows = json.loads(text)                       # one JSON array, as in dnli.zip
        except ValueError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]   # JSON lines
        with open(f"data/blind_external/pairs_dnli_{split}_full.jsonl", "w", encoding="utf-8") as f:
            for i, ex in enumerate(rows):
                old, new = tidy(ex["sentence2"]), tidy(ex["sentence1"])
                if len(old) < 8 or len(new) < 8:
                    continue
                # the Hugging Face copy has no "id" field (dnli.zip did): number by position
                f.write(json.dumps({"id": ex.get("id", f"{split}-{i}"), "source": f"dnli-{split}:{ex['label']}", "old_text": old,
                                    "new_text": new, "old_date": "2025-03-01", "new_date": "2025-06-01",
                                    "label": int(ex["label"] == "negative")}, ensure_ascii=False) + "\n")
        print(f"dnli {split}: {len(rows)} pairs", flush=True)

    def find_model(name: str, dataset: str | None = None) -> str:
        """A checkpoint folder under /kaggle/input, unpacking it first if the dataset
        holds it as a zip (an upload with --dir-mode zip may stay packed). A dataset
        that is itself the checkpoint (model.safetensors at its root, as
        primnox-memory-head-r6 is) is found by its slug."""
        found = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if os.path.isdir(p)]
        if found:
            return found[0]
        if dataset:
            weights = glob.glob(f"/kaggle/input/**/{dataset}/**/model.safetensors", recursive=True)
            if weights:
                return os.path.dirname(weights[0])
        for z in glob.glob(f"/kaggle/input/**/{name}*.zip", recursive=True):
            import zipfile
            zipfile.ZipFile(z).extractall(f"/tmp/{name}")
            inner = [p for p in glob.glob(f"/tmp/{name}/**/model.safetensors", recursive=True)]
            if inner:
                return os.path.dirname(inner[0])
        raise SystemExit(f"{name} not found; /kaggle/input holds: {sorted(glob.glob('/kaggle/input/*/*'))[:20]}")


    models = {"r4": find_model("laya-memory-r4"), "r6": find_model("laya-memory-r6", "primnox-memory-head-r6")}
    print(f"models: {models}", flush=True)
    files = ("data/blind_external/pairs_dnli_test_full.jsonl data/blind_external/pairs_dnli_verified_test_full.jsonl "
             "data/blind_external/pairs_lme_ku.jsonl")
    for tag, model in models.items():
        sh(f"python external_eval.py --model {model} --tag {tag}-kaggle --batch 64 --files {files}", check=False)
        sh(f"python external_eval.py --model {model} --tag {tag}-kaggle --sentences "
           f"--files data/blind_external/pairs_lme_ku.jsonl", check=False)
        sh(f"python blind_pairs.py --data data/lme_e2e/lme_e2e.json --model {model} --tag {tag}-kaggle --save-pairs",
           check=False)

    for f in glob.glob("results/*kaggle*"):
        if os.path.isfile(f):  # results/ also holds a kaggle_v2/ folder from an earlier round
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
