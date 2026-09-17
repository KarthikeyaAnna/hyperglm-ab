#!/usr/bin/env python3
"""Canonical AG graphs -> train / VAL / test task items + procedural graph.

Split discipline:
  * val = a deterministic 10 % of AG *train* videos, chosen by sha1(video_id) — independent of file order,
    reproducible anywhere, never overlaps test.
  * the procedural graph P (transition table shown in every prompt) is built from train-minus-val ONLY,
    so no val or test label reaches a prompt.
  * all tuning / early stopping reads val; test is for final numbers only.
Outputs: data/tasks/ag_{sga,rr}_{train,val,test}.jsonl, procedural_ag.json, data/splits/ag_split.json
(The [A]/[B] experiment uses SGA at F=0.9 and RR-hard; build_tasks_r13.py turns these into its own format.)
"""
import json, sys, argparse, collections, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import os as _os; DATA = Path(_os.environ.get("HYPERGLM_DATA", ROOT / "data"))   # data root (action_genome/, graphs/, tasks/, splits/)
from hyperglm_r import data, procedural, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--F", type=float, nargs="+", default=[0.9], help="SGA observed fractions; first is the headline")
ap.add_argument("--rr_per_video", type=int, default=2)
ap.add_argument("--val_frac", type=float, default=0.10)
ap.add_argument("--max_videos", type=int, default=None, help="debug subset per split")
ap.add_argument("--out", default=str(DATA / "tasks"))
ap.add_argument("--show_examples", action="store_true")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
SPLITS = DATA / "splits"; SPLITS.mkdir(parents=True, exist_ok=True)

def is_val(video_id, frac):
    return int(hashlib.sha1(video_id.encode()).hexdigest(), 16) % 10000 < int(round(frac * 10000))

train_all = data.load_videos(DATA / "graphs/ag_train.jsonl", a.max_videos)
test = data.load_videos(DATA / "graphs/ag_test.jsonl", a.max_videos)
val = collections.OrderedDict((v, f) for v, f in train_all.items() if is_val(v, a.val_frac))
train = collections.OrderedDict((v, f) for v, f in train_all.items() if not is_val(v, a.val_frac))
assert not (set(val) & set(train)) and not (set(val) & set(test)) and not (set(train) & set(test)), "split overlap"
print(f"videos: train {len(train)} | val {len(val)} ({100*len(val)/len(train_all):.1f}% of AG train) | test {len(test)}")

P = procedural.build(train)                     # train-minus-val only
procedural.save(P, OUT / "procedural_ag.json")
json.dump({"rule": f"val iff int(sha1(video_id),16) % 10000 < {int(round(a.val_frac*10000))}", "val_frac": a.val_frac,
           "procedural_built_from": "AG train minus val", "n_train": len(train), "n_val": len(val), "n_test": len(test),
           "val_videos": sorted(val), "train_videos": sorted(train)}, open(SPLITS / "ag_split.json", "w"), indent=1)

lens = collections.defaultdict(list); counts = {}
for split, vids in (("train", train), ("val", val), ("test", test)):
    per_video = a.rr_per_video if split == "train" else 1
    sga_specs = [(f"sga{F:.1f}".replace("sga0.9", "sga"), tasks.sga_items(vids, P, split, F=F)) for F in a.F]
    for task, gen in (*sga_specs,
                      ("rr", tasks.rr_items(vids, P, split, per_video=per_video, hard=True)),
                      ):
        n = 0
        with open(OUT / f"ag_{task}_{split}.jsonl", "w") as w:
            for item in gen:
                w.write(json.dumps(item) + "\n"); n += 1
                if split != "train": lens[(task, "target_B")].append(len(item["target_B"]))
        counts[(task, split)] = n
print("items:", {f"{t}_{s}": n for (t, s), n in counts.items()})
for k, v in lens.items(): print(f"  eval-split chars {k}: mean {sum(v)/len(v):.0f} max {max(v)}")

# ---- assertions: disjoint videos per task across splits; no masked-answer leak in val or test
tasknames = sorted({t for t, _ in counts})
for t in tasknames:
    vids = {s: {json.loads(l)["video"] for l in open(OUT / f"ag_{t}_{s}.jsonl")} for s in ("train", "val", "test")}
    assert not (vids["train"] & vids["val"]) and not (vids["train"] & vids["test"]) and not (vids["val"] & vids["test"]), f"video overlap in {t}"
print(f"  DISJOINT CHECK: train/val/test videos disjoint for {tasknames}")
for name in ("rr",):
    for split in ("val", "test"):
        leaks = n = 0
        for line in open(OUT / f"ag_{name}_{split}.jsonl"):
            it = json.loads(line); n += 1; obj, g = it["meta"]["object"], it["meta"]["group"]
            rows = [l.strip() for l in it["prompt_B"][1]["content"].split("transitions from last frame")[1].splitlines()]
            leaks += any(l.startswith(f"{obj}/{g}: {p} ->") for l in rows for p in it["gold"]["predicates"])
        print(f"  LEAK CHECK {name}_{split}: {leaks}/{n}"); assert leaks == 0
for F in a.F:
    name = f"sga{F:.1f}".replace("sga0.9", "sga")
    for split in ("val", "test"):
        ch = collections.Counter(json.loads(l)["n_change"] > 0 for l in open(OUT / f"ag_{name}_{split}.jsonl"))
        print(f"  {name}_{split}: items with >=1 changed slot {ch[True]}/{sum(ch.values())}")
if a.show_examples:
    ex = json.loads(open(OUT / "ag_sga_val.jsonl").readline()); print("\n=== EXAMPLE SGA val target_B ===\n" + ex["target_B"][:1200])
