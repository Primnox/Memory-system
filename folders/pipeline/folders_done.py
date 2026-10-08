"""How many people each set has fully filed.   python folders_done.py [set]  (exit 0 when that set is complete)"""
import json, re, sys
import folders
want = sys.argv[1] if len(sys.argv) > 1 else None
complete = True
for s in ("chat", "blind", "realtalk", "v3long"):
    done = part = tot = 0
    for sc in folders.load(s):
        f = folders.OUT / s / re.sub(r"[^\w\-]", "_", sc["scenario"]) / "state.json"
        tot += 1
        if f.exists():
            st = json.loads(f.read_text(encoding="utf-8"))
            done += len(st["facts"]) == len(sc["statements"])
            part += len(st["facts"]) != len(sc["statements"])
    print(f"{s}: {done} done, {part} part-way, of {tot}")
    if want in (s, "all") and done < tot:
        complete = False
sys.exit(0 if complete else 1)
