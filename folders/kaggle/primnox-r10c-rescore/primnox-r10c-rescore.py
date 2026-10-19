"""Kaggle GPU job: score round 10c's two models (r10c, r10c2) on the COMPLETE who-is-who sets.

Round 10c was scored on the first 9 blind people with writer + Gemini labels. The sets are now complete
(18 blind people with Claude Sonnet audits, 6 open-set people), and round 10d was scored on them. This runs the
same evaluation code (IDEVAL, shipped in primnox-identity-rows as primnox-r10.py) on the r10c models so r10c vs
r10d can be compared seed by seed on identical rows. Outputs identity_<tag>_<set>.txt in /kaggle/working.
"""
import ast
import glob
import os
import subprocess
import sys

subprocess.run("pip install -q laya", shell=True, check=True)
src = open(glob.glob("/kaggle/input/**/primnox-r10.py", recursive=True)[0], encoding="utf-8").read()
ideval = next(n.value.value for n in ast.parse(src).body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "IDEVAL")
open("/tmp/ideval.py", "w", encoding="utf-8").write(ideval)
for tag in ("r10c", "r10c2"):
    found = [p for p in glob.glob(f"/kaggle/input/**/laya-memory-{tag}", recursive=True) if os.path.isdir(p)]
    print(f"{tag}: {found}", flush=True)
    if found:
        subprocess.run([sys.executable, "/tmp/ideval.py", found[0], tag], check=False)
