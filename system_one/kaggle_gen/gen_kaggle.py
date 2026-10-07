"""Kaggle GPU job: write memory-head training people with two open model families.

Qwen 2.5 14B (Alibaba, Apache-2.0) and Mistral Nemo 12B (Mistral, Apache-2.0) are
served by Ollama on Kaggle's T4s, one server per GPU; each writes N people with
gen_people.py and the OTHER checks every label it gave. With one GPU, a single
server holds both and swaps between them. Outputs land in /kaggle/working for
`kaggle kernels output`.

(vLLM was the first choice; Kaggle's Python 3.13 has no torch for the vLLM that
still runs on a T4's compute 7.5, so Ollama, which does.)
"""
import glob
import json
import os
import subprocess
import time
import urllib.request

OUT = "/kaggle/working"
N = 20
QWEN, MISTRAL = "qwen2.5:14b", "mistral-nemo:12b"
GEN = glob.glob("/kaggle/input/**/gen_people.py", recursive=True)[0]
log = open(f"{OUT}/run.log", "a", buffering=1)


def say(msg: str) -> None:
    print(msg, flush=True)
    log.write(msg + "\n")


def sh(cmd: str, check: bool = True) -> None:
    say(f"$ {cmd}")
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tail -20 | tee -a {OUT}/run.log"], check=check)


def server(gpu: int | None, port: int) -> None:
    env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_NUM_PARALLEL="2",
               OLLAMA_MAX_LOADED_MODELS="2", OLLAMA_KEEP_ALIVE="2h")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.Popen(["ollama", "serve"], env=env, stdout=open(f"{OUT}/ollama_{port}.log", "w"),
                     stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=5)
            say(f"ollama up on port {port} (GPU {gpu})")
            return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"ollama on port {port} never came up")


def pull(port: int, model: str) -> None:
    sh(f"OLLAMA_HOST=127.0.0.1:{port} ollama pull {model} 2>&1 | grep -v '^pulling' | tail -2")


def gen(writer: str, checker: str, seed: int, out: str, name: str) -> subprocess.Popen:
    cmd = (f"PYTHONIOENCODING=utf-8 python {GEN} --kind train --writer {writer} --checker {checker} "
           f"-n {N} --workers 2 --seed {seed} --out {out} >> {OUT}/gen_{name}.log 2>&1")
    say(f"$ {cmd}")
    return subprocess.Popen(["bash", "-c", cmd])


t0 = time.time()
sh("curl -fsSL https://ollama.com/install.sh | sh")
gpus = int(subprocess.run(["bash", "-c", "nvidia-smi -L | wc -l"], capture_output=True, text=True).stdout or 0)
say(f"GPUs: {gpus}")
q_out, m_out = f"{OUT}/train_writer_qwen25.json", f"{OUT}/train_writer_mistralnemo.json"
if gpus >= 2:
    server(0, 11434), server(1, 11435)
    pull(11434, QWEN), pull(11435, MISTRAL)
    Q, M = f"ollama@11434:{QWEN}", f"ollama@11435:{MISTRAL}"
else:
    server(None, 11434)
    pull(11434, QWEN), pull(11434, MISTRAL)
    Q, M = f"ollama@11434:{QWEN}", f"ollama@11434:{MISTRAL}"
jobs = [gen(Q, M, 900, q_out, "qwen"), gen(M, Q, 910, m_out, "mistral")]
for j in jobs:
    j.wait()
for f in (q_out, m_out):
    n = len(json.load(open(f))) if os.path.exists(f) else 0
    say(f"{os.path.basename(f)}: {n} people")
for f in glob.glob(f"{OUT}/gen_*.log"):
    say(f"--- {os.path.basename(f)} (tail)")
    say("".join(open(f, encoding="utf-8", errors="replace").readlines()[-3:]))
say(f"done in {time.time() - t0:.0f}s")
