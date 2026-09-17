"""Load canonical per-frame graph JSONL (see scripts/build_ag_graphs.py) grouped by video."""
import json, collections

GROUPS = ("attention", "spatial", "contacting")


def load_videos(path, max_videos=None):
    """-> OrderedDict video_id -> list[frame_record] sorted by frame_index."""
    vids = collections.OrderedDict()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            vids.setdefault(r["video"], []).append(r)
            if max_videos and len(vids) > max_videos:
                vids.popitem(); break
    for v in vids.values():
        v.sort(key=lambda r: r["frame_index"])
    return vids


def frame_preds(frame):
    """-> {object_category: {group: [pred, ...]}} for one frame (subject is always the person)."""
    out = collections.OrderedDict()
    cats = {o["i"]: o["category"] for o in frame["objects"]}
    for t in frame["triplets"]:
        obj = cats[t["o"]]
        out.setdefault(obj, {g: [] for g in GROUPS})[t["type"]].append(t["pred"])
    return out
