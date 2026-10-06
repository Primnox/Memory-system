"""The generated-people validator and label handling, without calling any model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_people as g  # noqa: E402


def person(n=20, kind="train"):
    sts = [{"id": f"s{i}", "date": f"2025-{1 + i // 4:02d}-{1 + i % 28:02d}", "text": f"Statement number {i} about me.",
            "replaces": [], "relations": []} for i in range(1, n + 1)]
    for i in (5, 7, 9, 11, 13, 15):
        sts[i]["replaces"] = [f"s{i - 2}"]
    sts[17]["replaces"] = ["s2", "s3"]                       # a chain reaction
    for i, rel in zip((6, 8, 10, 12, 14, 16), ("adds", "detail", "other", "event", "adds", "other")):
        sts[i]["relations"] = [{"to": "s1", "relation": rel}]
    p = {"scenario": "x", "statements": sts}
    if kind == "test":
        p["today"] = "2026-01-10"
        p["questions"] = ([{"id": f"q{i}", "text": "?", "type": "current", "answer_ids": ["s19"], "as_of": None}
                           for i in range(1, 7)]
                          + [{"id": "q7", "text": "?", "type": "past", "answer_ids": ["s4"], "as_of": "2025-02-01"},
                             {"id": "q8", "text": "?", "type": "past", "answer_ids": ["s5"], "as_of": "2025-02-10"},
                             {"id": "q9", "text": "?", "type": "multi", "answer_ids": ["s1", "s7"], "as_of": None},
                             {"id": "q10", "text": "?", "type": "absent", "answer_ids": [], "as_of": None}])
    return p


def test_a_well_formed_person_has_no_problems():
    assert g.problems(person(), "train") == []
    assert g.problems(person(kind="test"), "test") == []


def test_ids_must_run_in_order():
    p = person()
    p["statements"][3]["id"] = "s9"
    assert any("out of order" in x for x in g.problems(p, "train"))


def test_dates_may_not_go_backwards():
    p = person()
    p["statements"][10]["date"] = "2024-01-01"
    assert any("backwards" in x for x in g.problems(p, "train"))


def test_a_label_may_only_point_back():
    p = person()
    p["statements"][5]["replaces"] = ["s19"]
    assert any("not an earlier message" in x for x in g.problems(p, "train"))


def test_a_pair_cannot_be_both_replaced_and_a_trap():
    p = person()
    p["statements"][5]["relations"] = [{"to": "s3", "relation": "detail"}]   # s6 already replaces s3
    assert any("both replaced and a trap" in x for x in g.problems(p, "train"))


def test_test_statements_must_fit_in_memory():
    p = person(kind="test")
    p["statements"][2]["text"] = "x" * (g.TEST_MAX_CHARS + 1)
    assert any("characters" in x for x in g.problems(p, "test"))
    assert g.problems(p, "train") == [] or not any("characters" in x for x in g.problems(p, "train"))


def test_a_past_question_needs_a_date_and_an_absent_one_no_answer():
    p = person(kind="test")
    p["questions"][6]["as_of"] = None
    p["questions"][9]["answer_ids"] = ["s1"]
    issues = g.problems(p, "test")
    assert any("past question without as_of" in x for x in issues)
    assert any("absent question with answers" in x for x in issues)


def test_too_few_updates_or_traps_is_refused():
    p = person()
    for s in p["statements"]:
        s["replaces"], s["relations"] = [], []
    issues = g.problems(p, "train")
    assert any("updates" in x for x in issues) and any("traps" in x for x in issues)


def test_labelled_pairs_lists_every_label():
    pairs = g.labelled_pairs(person())
    assert ("s2", "s18", True) in pairs and ("s3", "s18", True) in pairs
    assert ("s1", "s7", False) in pairs
    assert sum(1 for *_, rep in pairs if rep) == 8


def test_parse_json_reads_fenced_replies():
    assert g.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert g.parse_json('Sure! {"answers": ["yes"]} hope that helps') == {"answers": ["yes"]}
    with pytest.raises(ValueError):
        g.parse_json("no json here")


def test_disputed_training_labels_are_set_aside(monkeypatch):
    p = person()
    pairs = g.labelled_pairs(p)
    replies = ["no" if rep else "no" for *_, rep in pairs]      # disputes every replacement
    monkeypatch.setattr(g, "ask", lambda model, prompt, **kw: '{"answers": %s}' % str(replies).replace("'", '"'))
    args = type("A", (), {"kind": "train", "checker": "c", "judge": None})()
    audit = g.check(args, p)
    assert audit["disputed"] == 8
    assert ["s2", "s18"] in p["skip"]
    assert p["statements"][17]["replaces"] == ["s2", "s3"]      # training data keeps the writer's text


def test_a_judge_overturns_a_disputed_test_label(monkeypatch):
    p = person(kind="test")
    pairs = g.labelled_pairs(p)

    def judge_first_only(model, prompt, **kw):
        # the checker disputes every replacement; the judge overturns only the first
        if model == "checker":
            return '{"answers": %s}' % str(["no"] * len(pairs)).replace("'", '"')
        disputed = sum(1 for *_, rep in pairs if rep)
        return '{"answers": %s}' % str(["no"] + ["yes"] * (disputed - 1)).replace("'", '"')

    monkeypatch.setattr(g, "ask", judge_first_only)
    args = type("A", (), {"kind": "test", "checker": "checker", "judge": "judge"})()
    audit = g.check(args, p)
    assert audit["disputed"] == 8 and audit["overturned"] == 1
    first_rep_pair = next((a, b) for a, b, rep in pairs if rep)
    by_id = {s["id"]: s for s in p["statements"]}
    assert first_rep_pair[0] not in by_id[first_rep_pair[1]]["replaces"]


def test_build_train_leaves_out_disputed_pairs():
    import numpy as np
    import build_train
    p = person()
    p["skip"] = [["s3", "s8"]]                                   # s8 is a later message; s3 is a candidate for it
    vec = {s["text"]: np.ones(4) / 2 for s in p["statements"]}
    rows, stats = build_train.rows_for([p], vec)
    pairs = {(r["meta"]["old"], r["meta"]["new"]) for r in rows}
    assert ("s3", "s8") not in pairs and stats["skipped"] == 1
    assert ("s2", "s8") in pairs


def test_a_flawed_reply_is_sent_back_for_repair(monkeypatch):
    good = person()
    bad = json.loads(json.dumps(good))
    bad["statements"][12]["id"] = "s12"                         # a repeated id
    sent = []

    def fake(model, prompt, **kw):
        sent.append(prompt)
        return json.dumps(bad if len(sent) == 1 else good)

    monkeypatch.setattr(g, "ask", fake)
    args = type("A", (), {"kind": "train", "writer": "w", "style": "any"})()
    out = g.write_one(args, __import__("random").Random(1), 0)
    assert out is not None and len(sent) == 2
    assert "out of order" in sent[1] and '"s12"' in sent[1]     # the problems and the reply went back


def test_recheck_checks_people_already_written(monkeypatch, tmp_path):
    p = person()
    src = tmp_path / "raw.json"
    src.write_text(json.dumps([p]), encoding="utf-8")
    pairs = g.labelled_pairs(p)
    monkeypatch.setattr(g, "ask", lambda model, prompt, **kw: json.dumps(
        {"answers": ["yes" if rep else "no" for *_, rep in pairs]}))
    out = tmp_path / "checked.json"
    monkeypatch.setattr(sys, "argv", ["gen_people.py", "--kind", "train", "--writer", "vllm:8000:w",
                                      "--checker", "vllm:8001:c", "--recheck", str(src), "--out", str(out)])
    assert g.main() == 0
    (checked,) = json.loads(out.read_text(encoding="utf-8"))
    assert checked["scenario"] == "x" and "skip" not in checked      # the checker agreed with every label
