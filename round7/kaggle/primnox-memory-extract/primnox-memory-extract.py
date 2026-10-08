"""Kaggle GPU job: does LLM fact extraction fix the memory head on LongMemEval?

LongMemEval knowledge-update pairs (421; 72 real updates), each side a whole user
message. The head (rounds 4 and 6) scores every (old piece, new piece) combination
and keeps the highest, exactly as external_eval.py --sentences does; only the
pieces differ:
  whole      the message as it is
  rules      extract.py's sentence splitter (the current pipeline)
  llm        facts a Gemini model wrote down from each message (attached dataset
             primnox-lme-extract: {message: [facts]}, generic note-taker prompt)
Nothing here was trained on LongMemEval.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

OUT, SRC = "/kaggle/working", "/tmp/ms"
PAIRS = "data/blind_external/pairs_lme_ku.jsonl"


def sh(cmd: str, check: bool = True) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log"], check=check)


def find_model(name: str, dataset: str | None = None) -> str:
    found = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if os.path.isdir(p)]
    if found:
        return found[0]
    if dataset:
        weights = glob.glob(f"/kaggle/input/**/{dataset}/**/model.safetensors", recursive=True)
        if weights:
            return os.path.dirname(weights[0])
    raise SystemExit(f"{name} not found; /kaggle/input holds: {sorted(glob.glob('/kaggle/input/*/*/*'))[:20]}")


LLM_RUNNER = '''
import json, sys
import external_eval
facts = json.load(open(sys.argv[1], encoding="utf-8"))
missing = 0
def pieces(text, keep_all_if_none=True):
    global missing
    got = facts.get(text)
    if got is None:
        missing += 1
        return [text]
    return got or [text]   # no fact found: compare the whole message, as the rules do
external_eval.statements = pieces
sys.argv = ["external_eval.py"] + sys.argv[2:]
external_eval.main()
print(f"messages not in the facts file: {missing}", flush=True)
'''


def main() -> None:
    t0 = time.time()
    os.environ["PYTHONUNBUFFERED"] = "1"
    sh("pip install -q laya scikit-learn")
    sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}")
    os.chdir(f"{SRC}/system_one")
    facts = glob.glob("/kaggle/input/**/facts_lme_ku.json", recursive=True)
    assert facts, "facts_lme_ku.json not found in the attached datasets"
    with open("run_llm_pieces.py", "w") as f:
        f.write(LLM_RUNNER)
    models = {"r4": find_model("laya-memory-r4"), "r6": find_model("laya-memory-r6", "primnox-memory-head-r6")}
    print(f"models: {models}", flush=True)
    for tag, model in models.items():
        sh(f"python external_eval.py --model {model} --tag {tag}-whole --files {PAIRS}", check=False)
        sh(f"python external_eval.py --model {model} --tag {tag}-rules --sentences --files {PAIRS}", check=False)
        sh(f"python run_llm_pieces.py {facts[0]} --model {model} --tag {tag}-llm --sentences --files {PAIRS}",
           check=False)
    for f in glob.glob("results/external_*"):
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
