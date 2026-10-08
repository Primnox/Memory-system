"""Round 7's training mix: everything at once.   python build_mix.py

Writes Memory-system training files (system_one/data/train_*.json format:
{scenario, statements[id,date,text,replaces,(relations)], skip}) into ../kaggle_r7_ds/:

  clean   the existing generated people (writers A-D, DeepSeek), v3 (voted), the
          chat people's answer-key facts (voted) and the mixed-language people (voted)
  noisy   two normal-strength copies and one heavy copy of all of the above (noisy.py),
          seeds 101-103 — never the test copies' seeds (11, 12)

Labels never change under noise; disputed pairs stay in `skip`.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from noisy import mess_people

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "kaggle_r7_ds"
MS_DATA = HERE.parent / "ms" / "system_one" / "data"
V3 = HERE.parent / "kaggle_ds"
NOISE = [(101, 1.0), (102, 1.0), (103, 1.5)]
TEST_SEEDS = {11, 12}


def statements_only(sc: dict) -> dict:
    keep = {k: sc[k] for k in ("scenario", "skip", "writer", "style") if k in sc}
    keep["statements"] = [{k: s[k] for k in ("id", "date", "text", "replaces", "relations") if k in s}
                          for s in sc["statements"]]
    return keep


def load_sets() -> dict[str, list[dict]]:
    sets: dict[str, list[dict]] = {}
    sets["existing"] = [sc for f in sorted(MS_DATA.glob("train_writer_*.json"))
                        for sc in json.loads(f.read_text(encoding="utf-8"))]
    sets["v3"] = [sc for f in sorted(V3.glob("train_v3_*.json")) for sc in json.loads(f.read_text(encoding="utf-8"))]
    sets["chat"] = [json.loads(f.read_text(encoding="utf-8"))[0] for f in sorted((HERE / "chat").glob("p*/voted.json"))]
    sets["multilang"] = [json.loads(f.read_text(encoding="utf-8"))[0]
                         for f in sorted((HERE / "cases" / "multilang").glob("p*/voted.json"))]
    return {k: [statements_only(sc) for sc in v] for k, v in sets.items()}


def main() -> None:
    assert not TEST_SEEDS & {s for s, _ in NOISE}, "training noise must not reuse the test copies' seeds"
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("train_*.json"):
        old.unlink()
    sets = load_sets()
    stats = Counter()
    for name, scs in sets.items():
        if not scs:
            print(f"{name}: none yet")
            continue
        (OUT / f"train_r7_{name}.json").write_text(json.dumps(scs, ensure_ascii=False) + "\n", encoding="utf-8")
        stats["people clean"] += len(scs)
        stats["updates clean"] += sum(len(s["replaces"]) for sc in scs for s in sc["statements"])
        for seed, level in NOISE:
            noisy = mess_people(scs, seed=seed, copies=1, level=level)
            for sc in noisy:
                sc["scenario"] = f"{sc['scenario']}-n{seed}"
            (OUT / f"train_r7_{name}_noisy{seed}.json").write_text(json.dumps(noisy, ensure_ascii=False) + "\n",
                                                                   encoding="utf-8")
            stats["people noisy"] += len(noisy)
        print(f"{name}: {len(scs)} people, {sum(len(sc['statements']) for sc in scs)} statements, "
              f"{sum(len(s['replaces']) for sc in scs for s in sc['statements'])} updates, "
              f"{sum(len(sc.get('skip') or []) for sc in scs)} disputed pairs skipped")
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "primnox-memory-r7-train", "id": "anikethpani/primnox-memory-r7-train",
         "licenses": [{"name": "other"}]}), encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.glob("train_*.json"))
    print(f"total: {stats['people clean']} clean people ({stats['updates clean']} updates) + "
          f"{stats['people noisy']} noisy copies; {size / 1e6:.1f} MB in {OUT.name}")


if __name__ == "__main__":
    main()
