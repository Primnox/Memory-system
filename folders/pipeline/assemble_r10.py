"""Assemble round 10's who-is-who + duo inputs into ../kaggle_identity_ds.   python assemble_r10.py

Training = pilot + train people (writers Gemini Pro / Flash, cases from training sections only).
Evaluation only = blind (Sonnet writer, all sections) and open (GPT-OSS, no case list).
Run build_identity_rows.py for every set first; duo.py vote writes train_duo.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE.parent / "kaggle_identity_ds"


def main() -> None:
    rows, scen = [], []
    for s in ("pilot", "train"):
        f = DS / f"{s}_identity_rows.jsonl"
        if f.exists():
            rows += f.read_text(encoding="utf-8").splitlines()
            scen += json.loads((DS / f"train_identity_{s}.json").read_text(encoding="utf-8"))
    eval_names = {json.loads(l)["meta"]["scenario"] for s in ("blind", "open") if (DS / f"{s}_identity_rows.jsonl").exists()
                  for l in (DS / f"{s}_identity_rows.jsonl").read_text(encoding="utf-8").splitlines()}
    train_names = {sc["scenario"] for sc in scen}
    assert not eval_names & train_names, f"eval people in training: {eval_names & train_names}"
    # "fit_" names: the per-set files (pilot_..., train_...) stay; the kernel reads only these two
    (DS / "fit_identity_rows.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (DS / "train_identity_fit.json").write_text(json.dumps(scen, indent=1, ensure_ascii=False), encoding="utf-8")
    duo = json.loads((DS / "train_duo.json").read_text(encoding="utf-8")) if (DS / "train_duo.json").exists() else []
    print(f"training: {len(rows)} who-is-who rows from {len(scen)} people; {len(duo)} duo pairs; "
          f"eval people {len(eval_names)}")
    print(sorted(p.name for p in DS.iterdir()))


if __name__ == "__main__":
    main()
