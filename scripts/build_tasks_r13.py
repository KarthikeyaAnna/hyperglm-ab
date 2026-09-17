#!/usr/bin/env python3
"""Pilot-13 task files (DESIGN.md). Same split and procedural graph (train minus val), new SGA format.
Builds:
  ag_sga13_{train,val,test}.jsonl  slot timelines + changes-only answer format
  ag_rr_{train,val,test}.jsonl, procedural_ag.json  copied from data/tasks (unchanged)
Checks: split disjointness, lossless changes<->future round trip, no future leak."""
import sys, json, argparse, shutil, collections, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
DATA = Path(os.environ.get("HYPERGLM_DATA", ROOT / "data"))
from hyperglm_r import data, procedural, r13
from hyperglm_r.data import GROUPS, frame_preds

ap = argparse.ArgumentParser()
ap.add_argument("--train_F", type=float, nargs="+", default=[0.9, 0.7, 0.5])
ap.add_argument("--out", default=str(DATA / "tasks_r13"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); V2 = DATA / "tasks"

split = json.load(open(DATA / "splits/ag_split.json"))
P = procedural.load(V2 / "procedural_ag.json")
train_all = data.load_videos(DATA / "graphs/ag_train.jsonl"); test = data.load_videos(DATA / "graphs/ag_test.jsonl")
tr_ids, va_ids = set(split["train_videos"]), set(split["val_videos"])
assert not tr_ids & va_ids and not (tr_ids | va_ids) & set(test), "split overlap"
train = {v: f for v, f in train_all.items() if v in tr_ids}; val = {v: f for v, f in train_all.items() if v in va_ids}
print(f"videos train {len(train)} val {len(val)} test {len(test)}")


def check_item(it):
    # lossless: the gold changes rebuild the gold future exactly
    ok, ch, bad = r13.parse_changes(it["target_B"]); assert ok and bad == 0, it["id"]
    assert r13.apply_changes(it["gold"], it["observed"], ch) == {"future": {f: {o: {g: sorted(gs[g]) for g in GROUPS} for o, gs in c.items()} for f, c in it["gold"]["future"].items()}}, f"round trip {it['id']}"
    ok_a, ch_a, _ = r13.parse_changes(it["target_A"]); assert ok_a and ch_a == ch
    # no future leak: timelines/prompt never mention a frame after the last observed one except in the future-frame list
    last = max(int(f) for f in it["observed"]); user = it["prompt_B"][1]["content"]
    body = user.split("future frames to predict")[0]
    for f in it["meta"]["future_frames"]:
        assert f"@{f}" not in body and f"-{f} " not in body and f"-{f})" not in body, f"future frame {f} in timeline {it['id']}"
    assert all(int(f) > last for f in it["gold"]["future"])


def write(name, items):
    n = 0
    with open(OUT / name, "w") as fh:
        for it in items:
            check_item(it); fh.write(json.dumps(it) + "\n"); n += 1
    print(f"wrote {OUT / name}: {n}")
    return n


def gen_train():
    for F in a.train_F:
        yield from r13.sga13_items(train, P, "train", F)


write("ag_sga13_train.jsonl", gen_train())
for sp, vids in (("val", val), ("test", test)):
    items = list(r13.sga13_items(vids, P, sp, 0.9))
    v2 = {json.loads(l)["video"]: json.loads(l) for l in open(V2 / f"ag_sga_{sp}.jsonl")}
    assert {it["video"] for it in items} == set(v2), f"{sp}: item set differs from v2"
    for it in items:
        assert it["gold"] == v2[it["video"]]["gold"] and it["observed"] == v2[it["video"]]["observed"], f"{sp} gold/observed differs from v2: {it['id']}"
    write(f"ag_sga13_{sp}.jsonl", items)

for f in ("ag_rr_train.jsonl", "ag_rr_val.jsonl", "ag_rr_test.jsonl", "procedural_ag.json"):
    shutil.copy(V2 / f, OUT / f)

# RR priors from train frames only
c1, c3 = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
for frames in train.values():
    for fr in frames:
        for obj, gs in frame_preds(fr).items():
            for g in GROUPS:
                if not gs[g]: continue
                others = "|".join(" ".join(sorted(gs[g2])) for g2 in GROUPS if g2 != g)
                gold = " ".join(sorted(gs[g]))
                c1[f"{obj}/{g}"][gold] += 1; c3[f"{obj}/{g}/{others}"][gold] += 1
json.dump({"object_group": {k: v.most_common(1)[0][0] for k, v in c1.items()},
           "object_group_others": {k: v.most_common(1)[0][0] for k, v in c3.items() if sum(v.values()) >= 3},
           "built_from": "AG train minus val (frames)"}, open(OUT / "rr_priors.json", "w"))
print("wrote rr_priors.json; all checks passed")
