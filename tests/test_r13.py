#!/usr/bin/env python3
"""Unit checks for the pilot-13 format and decoding. Self-contained, no data files.
Run: python tests/test_r13.py   (or: python -m pytest tests/)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperglm_r import metrics, r13

GROUPS = ("attention", "spatial", "contacting")
OBSERVED = {"30": [["cup", "attention", "looking_at"], ["cup", "spatial", "in_front_of"], ["cup", "contacting", "touching"]]}
GOLD = {"future": {
    "40": {"cup": {"attention": ["looking_at"], "spatial": ["in_front_of"], "contacting": ["touching"]}},
    "50": {"cup": {"attention": ["looking_at"], "spatial": ["in_front_of"], "contacting": ["holding"]}},
    "60": {"cup": {"attention": ["not_looking_at"], "spatial": ["in_front_of"], "contacting": ["holding"]}}}}
CHANGES = [["cup", "contacting", ["holding"], 50], ["cup", "attention", ["not_looking_at"], 60]]
IT = {"gold": GOLD, "observed": OBSERVED}
ANSWER = '<infer>\ncup/contacting: touching for 3 frames -> holding from 50\n</infer>\n<answer>{"changes":[["cup","contacting",["holding"],50],["cup","attention",["not_looking_at"],60]]}</answer>'
EMPTY = '<answer>{"changes":[]}</answer>'


def test_changes_rebuild_the_gold_future():
    assert r13.apply_changes(GOLD, OBSERVED, CHANGES) == GOLD

def test_parse_answer_and_reject_garbage():
    ok, ch, bad = r13.parse_changes(ANSWER)
    assert ok and bad == 0 and ch == CHANGES
    assert not r13.parse_changes("it stays")[0]
    ok, ch, bad = r13.parse_changes('<answer>{"changes":[["cup","nonsense",["x"],50],["cup","contacting","holding",50]]}</answer>')
    assert ok and bad == 1 and ch == [["cup", "contacting", ["holding"], 50]]     # bad group dropped, bare string accepted

def test_empty_answer_is_persistence():
    dec = r13.apply_changes(GOLD, OBSERVED, [])
    assert dec == metrics.persist_prediction(GOLD, OBSERVED)

def test_votes_and_threshold():
    vote, n = r13.votes(IT, [ANSWER, EMPTY, EMPTY, EMPTY])
    assert n == 4 and vote[("50", "cup", "contacting", "holding")] == 0.25
    low = r13.decide(IT, vote, tau=0.25); high = r13.decide(IT, vote, tau=0.5)
    assert metrics.transition_scores(GOLD, low, OBSERVED)["f1"] == 1.0          # 1 of 4 samples is enough at tau 0.25
    assert metrics.transition_scores(GOLD, high, OBSERVED)["f1"] == 0.0         # not at tau 0.5
    assert "touching" in high["future"]["50"]["cup"]["contacting"]              # current state kept (3/4 votes)

def test_unparsed_samples_abstain():
    vote, n = r13.votes(IT, [ANSWER, "garbage", "garbage", "garbage"])
    assert n == 1 and vote[("50", "cup", "contacting", "holding")] == 1.0

def test_ranked_list_is_scored_by_rank():
    vote, _ = r13.votes(IT, [ANSWER])
    P = {k: {g: {} for g in GROUPS} for k in ("global", "stay", "cond", "cond_stay")}
    ranked = r13.ranked(IT, vote, P)
    assert metrics.sga_scores(GOLD, ranked, ks=(10,))["R@10"] == 1.0
    assert metrics.sga_scores(GOLD, {"future": {f: lst[:1] for f, lst in ranked["future"].items()}}, ks=(10,))["R@10"] < 1.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
