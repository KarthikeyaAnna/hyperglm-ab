#!/usr/bin/env python3
"""Action Genome -> canonical per-frame scene-graph JSONL (same schema as build_vsgr_graphs.py).

Inputs  data/action_genome/annotations/{object_bbox_and_relationship.pkl, person_bbox.pkl,
        object_classes.txt, relationship_classes.txt}
Outputs data/graphs/ag_{train,test}.jsonl, data/graphs/ag_vocab.json, data/graphs/ag_stats.json

AG facts encoded here: one person per frame (from person_bbox.pkl, xyxy, detector output at
480x270); at most one instance of each object class per frame, so track_id := class index+1
within a video and the person is track 0. Every triplet is person -> object with one of three
typed predicate groups (attention / spatial / contacting); an object may carry several
predicates per group per frame. Only visible objects with a bbox are kept. Split comes from
metadata.set. No video filtering here — SceneSayer's "drop videos with < 3 annotated frames"
is applied at task-construction time so the canonical file stays complete.
"""
import pickle, json, collections, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
import os as _os; DATA = Path(_os.environ.get("HYPERGLM_DATA", ROOT / "data"))   # data root (action_genome/, graphs/, tasks/, splits/)
A, OUT = DATA / "action_genome/annotations", DATA / "graphs"
obj_classes = [l.strip() for l in open(A / "object_classes.txt") if l.strip()]
rel_classes = [l.strip() for l in open(A / "relationship_classes.txt") if l.strip()]
# AG's txt files strip underscores/spaces; the pkl uses underscored names. Map by removing '_'.
GROUPS = {"attention_relationship": "attention", "spatial_relationship": "spatial", "contacting_relationship": "contacting"}
D = pickle.load(open(A / "object_bbox_and_relationship.pkl", "rb"))
P = pickle.load(open(A / "person_bbox.pkl", "rb"))

# bbox convention check: in xyxy, bbox[2] >= bbox[0] always; in xywh it often is not.
viol = tot = 0
for objs in list(D.values())[:20000]:
    for o in objs:
        if o["visible"] and o["bbox"] is not None:
            tot += 1; viol += o["bbox"][2] < o["bbox"][0]
obj_fmt = "xywh" if viol / max(1, tot) > 0.05 else "xyxy"
print(f"[bbox] object boxes: {100*viol/max(1,tot):.1f}% have bbox[2]<bbox[0] -> treating as {obj_fmt}; person boxes are xyxy (per pkl)")

by_video = collections.defaultdict(list)
for fr in D: v, fn = fr.split("/"); by_video[v].append((int(fn.split(".")[0]), fr))
writers = {s: open(OUT / f"ag_{s}.jsonl", "w") for s in ("train", "test")}
stats = collections.defaultdict(collections.Counter); predset = collections.defaultdict(set)
cls_index = {c: i for i, c in enumerate(sorted({o["class"] for objs in D.values() for o in objs}))}
for v, frames in by_video.items():
    frames.sort()
    for fi, fr in frames:
        objs = D[fr]; split = objs[0]["metadata"]["set"] if objs else None
        pb = P.get(fr, {}); pbox = pb.get("bbox"); pbox = [float(x) for x in pbox[0]] if pbox is not None and len(pbox) else None
        W, H = (pb.get("bbox_size") or (480, 270))
        objects = [{"i": 0, "track_id": 0, "category": "person", "bbox": pbox}]
        triplets = []
        for o in objs:
            if not o["visible"] or o["bbox"] is None: continue
            idx = len(objects)
            objects.append({"i": idx, "track_id": 1 + cls_index[o["class"]], "category": o["class"], "bbox": [float(x) for x in o["bbox"]]})
            for g, ty in GROUPS.items():
                for p in (o.get(g) or []):
                    triplets.append({"s": 0, "o": idx, "pred": p, "type": ty}); predset[ty].add(p)
        if split is None or len(objects) == 1 and not triplets: continue
        rec = {"source": "ag", "split": split, "video_id": v, "video": v, "frame_id": fr, "frame_index": fi,
               "file_name": fr, "width": int(W), "height": int(H), "bbox_format": obj_fmt, "person_bbox_format": "xyxy",
               "objects": objects, "triplets": triplets, "attributes": [], "text": None}
        writers[split].write(json.dumps(rec) + "\n")
        st = stats[split]; st["frames"] += 1; st["objects"] += len(objects) - 1; st["triplets"] += len(triplets)
        st["frames_with_person_box"] += pbox is not None
        for t in triplets: st[f"triplets_{t['type']}"] += 1
    stats[objs[0]["metadata"]["set"]]["videos"] += 1
    stats[objs[0]["metadata"]["set"]][f"videos_with_>=3_frames"] += len(frames) >= 3
for w in writers.values(): w.close()
vocab = {"objects": ["person"] + sorted(cls_index), **{ty: sorted(s) for ty, s in predset.items()}}
json.dump(vocab, open(OUT / "ag_vocab.json", "w"), indent=1)
json.dump({k: dict(v) for k, v in stats.items()}, open(OUT / "ag_stats.json", "w"), indent=1)
for k, v in stats.items(): print(f"ag_{k}: " + " | ".join(f"{a}={b}" for a, b in sorted(v.items())))
print("vocab:", {k: len(v) for k, v in vocab.items()})
