"""The memory head on any DirectX 12 GPU (AMD, Intel, NVIDIA) through ONNX Runtime's
DirectML provider. Laya's own ONNX agent only tries CUDA, then CPU.

    agent = onnx_dml.load("kaggle_out4/models/laya-memory-r4", "models/onnx/laya-memory-r4.onnx")
    agent.predict_batch(states, QUESTIONS)   # same API as laya.agent.Agent

Export once with laya_export_onnx_upstream.py (fp32; int8 is CPU-only and costs accuracy).
`python onnx_dml.py --check` compares it with the PyTorch model on development pairs.
"""
from __future__ import annotations

import argparse
import time


def patch_for_dml(src: str, dst: str) -> int:
    """Copy `src` with every Reshape's `allowzero` set to 0, which DirectML requires
    (with 1 it fails "The parameter is incorrect"). The two differ only for a 0
    in the target shape, and this model's shapes (batch, sequence, 3, -1, 64...)
    never hold one. Returns how many nodes changed."""
    import os

    import onnx
    model = onnx.load(src)
    changed = 0
    for node in model.graph.node:
        if node.op_type == "Reshape":
            for attr in node.attribute:
                if attr.name == "allowzero" and attr.i == 1:
                    attr.i = 0
                    changed += 1
    onnx.save(model, dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(dst) + ".data")
    return changed


def load(model_dir: str, onnx_path: str, provider: str = "DmlExecutionProvider"):
    import onnxruntime as ort
    from laya.onnx_agent import ONNXAgent
    agent = ONNXAgent(model_dir, onnx_path=onnx_path)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # DirectML needs both of these off
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    agent.session = ort.InferenceSession(onnx_path, sess_options=so, providers=[provider, "CPUExecutionProvider"])
    if agent.session.get_providers()[0] != provider:
        raise RuntimeError(f"{provider} did not initialise; got {agent.session.get_providers()}")
    return agent


def check(model_dir: str, onnx_path: str, n: int) -> None:
    from laya.agent import Agent
    from build_train import QUESTIONS
    from screen import load_pairs, state_text
    states = [state_text(p) for p in load_pairs()[:n]]

    def run(agent, label):
        agent.predict_batch(states[:4], QUESTIONS)      # warm-up
        t = time.perf_counter()
        out = [r["answers"]["relation"]["probabilities"]["replaces"] for r in agent.predict_batch(states, QUESTIONS)]
        ms = (time.perf_counter() - t) * 1000 / len(states)
        print(f"{label}: {ms:.1f} ms per comparison")
        return out

    gpu = run(load(model_dir, onnx_path), "onnx + DirectML")
    cpu = run(Agent(model_dir), "pytorch CPU")
    diffs = [abs(a - b) for a, b in zip(gpu, cpu)]
    flips = sum((a >= 0.5) != (b >= 0.5) for a, b in zip(gpu, cpu))
    print(f"{len(states)} pairs: max |dP| {max(diffs):.4f}, mean {sum(diffs) / len(diffs):.5f}, "
          f"decisions that differ at 0.5: {flips}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kaggle_out4/models/laya-memory-r4")
    ap.add_argument("--onnx", default="models/onnx/laya-memory-r4.onnx")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("-n", type=int, default=120)
    ap.add_argument("--patch", metavar="SRC", help="write --onnx as SRC made DirectML-compatible")
    args = ap.parse_args()
    if args.patch:
        print(f"{patch_for_dml(args.patch, args.onnx)} Reshape nodes changed")
    if args.check:
        check(args.model, args.onnx, args.n)
