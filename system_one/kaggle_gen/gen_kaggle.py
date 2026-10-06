"""Kaggle GPU job: write memory-head training people with two open model families.

Qwen 2.5 14B (Alibaba) and Mistral Nemo 12B (Mistral) are served with vLLM on
Kaggle's T4s; each writes N people with gen_people.py, and the OTHER checks every
label it gave. With two GPUs both run at once; with one, they take turns (write
without checking, then the other model rechecks). Outputs land in
/kaggle/working for `kaggle kernels output`.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

OUT = "/kaggle/working"
N = 20
KIT = os.path.dirname(glob.glob("/kaggle/input/**/gen_people.py", recursive=True)[0])
GEN = os.path.join(KIT, "gen_people.py")
CANDIDATES = {
    "qwen": ["Qwen/Qwen2.5-14B-Instruct-AWQ"],
    "mistral": ["casperhansen/mistral-nemo-instruct-2407-awq", "solidrust/Mistral-Nemo-Instruct-2407-AWQ",
                "Qwen/Qwen2.5-7B-Instruct-AWQ"],
}
log = open(f"{OUT}/run.log", "a", buffering=1)


def say(msg: str) -> None:
    print(msg, flush=True)
    log.write(msg + "\n")


def sh(cmd: str, check: bool = True) -> None:
    say(f"$ {cmd}")
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log"], check=check)


def pick(family: str) -> str:
    from huggingface_hub import model_info
    for repo in CANDIDATES[family]:
        try:
            model_info(repo)
            return repo
        except Exception as e:
            say(f"{repo}: unavailable ({type(e).__name__})")
    raise SystemExit(f"no {family} model available")


def serve(model: str, gpu: int, port: int) -> subprocess.Popen:
    args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
            "--served-model-name", model, "--dtype", "half", "--max-model-len", "10240",
            "--gpu-memory-utilization", "0.92", "--port", str(port), "--disable-log-requests"]
    if "awq" in model.lower():
        args += ["--quantization", "awq"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    proc = subprocess.Popen(args, env=env, stdout=open(f"{OUT}/vllm_{port}.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            say(f"{model} up on GPU {gpu}, port {port}")
            return proc
        except Exception:
            if proc.poll() is not None:
                raise SystemExit(f"vLLM for {model} exited; see vllm_{port}.log")
            time.sleep(10)
    raise SystemExit(f"vLLM for {model} never came up")


def gen(writer: str, checker: str, seed: int, out: str, extra: str = "") -> None:
    sh(f"PYTHONIOENCODING=utf-8 python {GEN} --kind train --writer {writer} --checker {checker} "
       f"-n {N} --workers 6 --seed {seed} --out {out} {extra}", check=False)


t0 = time.time()
sh("pip install -q vllm==0.6.3.post1")
import torch  # noqa: E402  (after vLLM pulls its own torch)
gpus = torch.cuda.device_count()
say(f"GPUs: {gpus} x {torch.cuda.get_device_name(0) if gpus else 'none'}")
qwen, mistral = pick("qwen"), pick("mistral")
Q, M = f"vllm:8000:{qwen}", f"vllm:8001:{mistral}"
q_out, m_out = f"{OUT}/train_writer_qwen25.json", f"{OUT}/train_writer_mistralnemo.json"

if gpus >= 2:
    a, b = serve(qwen, 0, 8000), serve(mistral, 1, 8001)
    p1 = subprocess.Popen(["bash", "-c", f"PYTHONIOENCODING=utf-8 python {GEN} --kind train --writer {Q} "
                           f"--checker {M} -n {N} --workers 6 --seed 900 --out {q_out} >> {OUT}/gen_qwen.log 2>&1"])
    p2 = subprocess.Popen(["bash", "-c", f"PYTHONIOENCODING=utf-8 python {GEN} --kind train --writer {M} "
                           f"--checker {Q} -n {N} --workers 6 --seed 910 --out {m_out} >> {OUT}/gen_mistral.log 2>&1"])
    p1.wait(), p2.wait()
    a.terminate(), b.terminate()
else:
    a = serve(qwen, 0, 8000)
    gen(Q, M, 900, f"{OUT}/raw_qwen.json", "--no-check")
    a.terminate(); a.wait(); time.sleep(10)
    b = serve(mistral, 0, 8001)
    gen(M, Q, 910, f"{OUT}/raw_mistral.json", "--no-check")
    gen(Q, M, 0, q_out, f"--recheck {OUT}/raw_qwen.json")
    b.terminate(); b.wait(); time.sleep(10)
    a = serve(qwen, 0, 8000)
    gen(M, Q, 0, m_out, f"--recheck {OUT}/raw_mistral.json")
    a.terminate()

for f in (q_out, m_out):
    n = len(json.load(open(f))) if os.path.exists(f) else 0
    say(f"{os.path.basename(f)}: {n} people")
say(f"done in {time.time() - t0:.0f}s")
