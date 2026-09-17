"""Procedural graph P (HyperGLM Eq. 3-4):
- P_global[type][a][b]   : transition prob a->b (a != b), rows normalised   (the paper's P)
- stay[type][a]          : prob the predicate persists to the next annotated frame
- P_cond[type][obj][a][b]: same, conditioned on the object class, backing off to global
Built from TRAIN only. Consecutive *annotated* frames of the same (video, object) define a step.
"""
import json, collections
from .data import GROUPS, frame_preds


def build(videos, min_count=5):
    trans = {g: collections.defaultdict(collections.Counter) for g in GROUPS}
    stay = {g: collections.defaultdict(lambda: [0, 0]) for g in GROUPS}            # a -> [stayed, total]
    ctrans = {g: collections.defaultdict(lambda: collections.defaultdict(collections.Counter)) for g in GROUPS}
    cstay = {g: collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0])) for g in GROUPS}
    for frames in videos.values():
        prev = None
        for fr in frames:
            cur = frame_preds(fr)
            if prev is not None:
                for obj, gs in cur.items():
                    if obj not in prev: continue
                    for g in GROUPS:
                        A, B = set(prev[obj][g]), set(gs[g])
                        for a in A:
                            stay[g][a][1] += 1; cstay[g][obj][a][1] += 1
                            if a in B:
                                stay[g][a][0] += 1; cstay[g][obj][a][0] += 1
                            for b in B - {a}:
                                trans[g][a][b] += 1; ctrans[g][obj][a][b] += 1
            prev = cur

    def norm(counter):
        tot = sum(counter.values())
        return {b: round(c / tot, 4) for b, c in counter.most_common() if c / tot >= 0.01} if tot else {}

    P = {"global": {}, "stay": {}, "cond": {}, "cond_stay": {}}
    for g in GROUPS:
        P["global"][g] = {a: norm(c) for a, c in trans[g].items()}
        P["stay"][g] = {a: round(s / t, 4) for a, (s, t) in stay[g].items() if t}
        P["cond"][g] = {obj: {a: norm(c) for a, c in d.items() if sum(c.values()) >= min_count} for obj, d in ctrans[g].items()}
        P["cond_stay"][g] = {obj: {a: round(s / t, 4) for a, (s, t) in d.items() if t >= min_count} for obj, d in cstay[g].items()}
    return P


def successors(P, group, obj, pred, k=3):
    """-> (stay_prob, [(b, prob), ...]) conditioned on obj when available, else global."""
    row = P["cond"][group].get(obj, {}).get(pred) or P["global"][group].get(pred, {})
    st = P["cond_stay"][group].get(obj, {}).get(pred, P["stay"][group].get(pred, 0.0))
    return st, list(row.items())[:k]


def save(P, path):
    json.dump(P, open(path, "w"))


def load(path):
    return json.load(open(path))
