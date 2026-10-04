"""Fine-tune Laya on the memory-head rows (see SPEC_memory_head.md).

Laya's own recipe, unchanged (laya_finetune_upstream.py, Apache-2.0, from
github.com/NandhaKishorM/laya): RLCD plus cross-entropy, then temperature
calibration on a held-back slice. Only the data source differs: our JSONL
rows instead of the typed-decisions dataset.

Soft "not a replacement" rows far outnumber the rest, so at most
`--soft-ratio` of them are kept per labelled row (replacement or trap).

    system_one/.venv/Scripts/python system_one/train_laya.py --device cpu --epochs 3
then score it on the validation pairs:
    system_one/.venv/Scripts/python system_one/screen.py --scorers laya_ft \\
        --laya-ft system_one/models/laya-memory-v1 --tag v1
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

import laya_finetune_upstream as up

HERE = Path(__file__).resolve().parent


def load_rows(paths: list[Path], soft_ratio: float, seed: int) -> list[dict]:
    rows = [json.loads(line) for path in paths
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hard = [r for r in rows if r["meta"]["kind"] != "soft"]
    soft = [r for r in rows if r["meta"]["kind"] == "soft"]
    random.Random(seed).shuffle(soft)
    ours = sum(1 for r in hard if r["meta"].get("source") != "dnli")   # the cap follows our labelled rows only
    keep = soft[:int(ours * soft_ratio)] if soft_ratio > 0 else soft
    print("rows:", dict(Counter(r["meta"]["kind"] for r in hard + keep)), f"(soft kept {len(keep)} of {len(soft)})")
    return hard + keep


def prepare_items(model_dir: str, rows: list[dict], items_path: Path) -> None:
    cfg = json.loads((Path(model_dir) / "rl_agent_config.json").read_text())
    cfg = {**cfg, "max_len": cfg.get("max_len", 1024), "head_max_len": cfg.get("head_max_len", 256)}
    tok = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
    items, skipped = [], 0
    for row in rows:
        state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
        for qid, question in questions.items():
            item = up.build_training_item(tok, cfg, state, question, gold[qid])
            if item is None:
                skipped += 1
            else:
                items.append(item)
    items_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, items_path)
    print(f"{len(items)} training items -> {items_path} (skipped {skipped})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=[str(HERE / "data" / "train_rows.jsonl"),
                                                 str(HERE / "data" / "dnli_rows.jsonl")])
    ap.add_argument("--model-dir", default=str(HERE / "models" / "laya_base"))
    ap.add_argument("--output-dir", default=str(HERE / "models" / "laya-memory-v1"))
    ap.add_argument("--soft-ratio", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--calib-max", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--limit", type=int, help="use only the first N rows after balancing (timing pilot)")
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    model_dir = up.prepare_model(args.model_dir)
    rows = load_rows([Path(r) for r in args.rows if Path(r).exists()], args.soft_ratio, args.seed)
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    items_path = Path(args.output_dir) / "train_items.pt"
    prepare_items(model_dir, rows, items_path)
    up.train(args, model_dir, items_path, torch.device(args.device))


if __name__ == "__main__":
    main()
