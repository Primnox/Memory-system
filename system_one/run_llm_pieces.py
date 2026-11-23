
import json, sys
import external_eval
facts = json.load(open(sys.argv[1], encoding="utf-8"))
external_eval.statements = lambda text, keep_all_if_none=True: facts.get(text) or [text]
sys.argv = ["external_eval.py"] + sys.argv[2:]
external_eval.main()
