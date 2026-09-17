"""Non-LLM learned reference for SGA (DESIGN.md): gradient-boosted candidate scorer over graph features
(slot dwell, horizon, previous states, other groups, transition-table probabilities). Fit on AG train minus val
(cut points F 0.5-0.9), thresholds chosen on VAL, reported once on TEST with the pilot-13 scorer's metrics.
  python scripts/gbm_reference_r13.py --out runs/p13_reference/gbm.json"""
import json, sys, statistics as st, random, argparse, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from hyperglm_r import data, procedural, metrics, r13
from hyperglm_r.data import GROUPS, frame_preds

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); A_ = ap.parse_args()
D = Path(os.environ.get("HYPERGLM_DATA", ROOT / "data"))
split = json.load(open(f"{D}/splits/ag_split.json")); train_ids = set(split["train_videos"])
P = procedural.load(f"{D}/tasks/procedural_ag.json")
videos = data.load_videos(f"{D}/graphs/ag_train.jsonl")
train_v = {v: fr for v, fr in videos.items() if v in train_ids}
print("train videos", len(train_v))

VOC = {}
def vid_(x): return VOC.setdefault(x, len(VOC))

def slot_history(seq_preds, obj, g):
    """list over observed frames of the slot's predicate set (None if object absent)."""
    return [set(fp[obj][g]) if obj in fp else None for fp in seq_preds]

def features(obs_frames, fut_index, obj, g, b, k, n_future, N, seq):
    """one candidate b for slot (obj,g) at future step k (1-based)."""
    hist = slot_history(seq, obj, g)
    last = hist[-1] or set()
    a = sorted(last)[0] if last else "-"
    stay, nxt = procedural.successors(P, g, obj, a, 10) if last else (0.0, [])
    nxd = dict(nxt)
    # dwell of the current state
    dwell = 0
    for h in reversed(hist):
        if h is not None and h == last: dwell += 1
        else: break
    # history of b in this slot
    seen = [i for i, h in enumerate(hist) if h and b in h]
    since = (len(hist) - 1 - seen[-1]) if seen else -1
    frac_b = len(seen) / len(hist)
    changes = sum(1 for i in range(1, len(hist)) if hist[i] is not None and hist[i - 1] is not None and hist[i] != hist[i - 1])
    other = [sorted(seq[-1][obj][g2])[0] if seq[-1][obj][g2] else "-" for g2 in GROUPS if g2 != g]
    gap = int(fut_index) - obs_frames[-1]["frame_index"]
    rank = [x for x, _ in nxt].index(b) if b in nxd else 99
    return [GROUPS.index(g), vid_(obj), vid_(b), vid_(a), float(b in last), stay, nxd.get(b, 0.0), rank, dwell, since, frac_b,
            changes / max(1, len(hist) - 1), vid_(other[0]), vid_(other[1]), k, n_future, gap, len(obs_frames) / N, len(last)]

CAT = [0, 1, 2, 3, 12, 13]

def candidates(obs_frames, obj, g, seq):
    hist = slot_history(seq, obj, g); last = hist[-1] or set()
    c = set(last)
    for a in last:
        _, nxt = procedural.successors(P, g, obj, a, 5); c |= {b for b, _ in nxt}
    for h in hist:
        if h: c |= h
    return sorted(c)

def windows(frames, F):
    N = len(frames); n_obs = max(1, min(N - 1, int(round(F * N))))
    obs, fut = frames[:n_obs], frames[n_obs:]
    last = frame_preds(obs[-1])
    return obs, fut, list(last), N

def build_rows(vids, Fs, max_rows=None, rng=None):
    X, y = [], []
    for v, frames in vids.items():
        if len(frames) < 3: continue
        for F in Fs:
            obs, fut, objs, N = windows(frames, F)
            if not objs: continue
            seq = [frame_preds(fr) for fr in obs]
            for obj in objs:
                for g in GROUPS:
                    cands = candidates(obs, obj, g, seq)
                    for k, fr in enumerate(fut, 1):
                        fp = frame_preds(fr)
                        if obj not in fp: continue
                        gold = set(fp[obj][g])
                        for b in cands:
                            X.append(features(obs, fr["frame_index"], obj, g, b, k, len(fut), N, seq)); y.append(int(b in gold))
    return np.array(X, dtype=float), np.array(y)

X, y = build_rows(train_v, (0.5, 0.6, 0.7, 0.8, 0.9))
print("train rows", X.shape, "pos rate", y.mean().round(3))
clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, categorical_features=CAT, random_state=0)
clf.fit(X, y)

test_videos = data.load_videos(f"{D}/graphs/ag_test.jsonl")
ITEMS = {sp: [json.loads(l) for l in open(f"{D}/tasks_r13/ag_sga13_{sp}.jsonl")] for sp in ("val", "test")}
VIDS = {"val": {v: fr for v, fr in videos.items() if v in {it["video"] for it in ITEMS["val"]}}, "test": test_videos}


def scores_for(it, vids):
    frames = vids[it["video"]]; obs, fut, objs, N = windows(frames, it["F"])
    rows, keys = [], []; seq = [frame_preds(fr) for fr in obs]
    for obj in objs:
        for g in GROUPS:
            cands = candidates(obs, obj, g, seq)
            for k, fr in enumerate(fut, 1):
                if str(fr["frame_index"]) not in it["gold"]["future"]: continue
                for b in cands:
                    rows.append(features(obs, fr["frame_index"], obj, g, b, k, len(fut), N, seq)); keys.append((str(fr["frame_index"]), obj, g, b))
    p = clf.predict_proba(np.array(rows, dtype=float))[:, 1] if rows else []
    return {k: float(v) for k, v in zip(keys, p)}


SC = {sp: {it["id"]: scores_for(it, VIDS[sp]) for it in ITEMS[sp]} for sp in ITEMS}


def trans(it, sc, tk, tn): return metrics.transition_scores(it["gold"], r13.decide(it, sc, tn, keep=tk), it["observed"])


grid = [(tk, tn) for tk in (0.2, 0.3, 0.4, 0.5) for tn in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)]
val_f1 = {t: st.mean(trans(it, SC["val"][it["id"]], *t)["f1"] for it in ITEMS["val"]) for t in grid}
best = max(grid, key=lambda t: val_f1[t])
P13 = procedural.load(f"{D}/tasks_r13/procedural_ag.json")
out = {"thresholds_from_val": {"keep": best[0], "new": best[1]}, "val_transition_f1_macro": round(100 * val_f1[best], 1)}
for sp in ("val", "test"):
    its = ITEMS[sp]
    m = [trans(it, SC[sp][it["id"]], *best) for it in its]
    rule = [metrics.transition_scores(it["gold"], metrics.table_rule_prediction(it["gold"], it["observed"], P13, 0.75), it["observed"]) for it in its]
    ranked = [{"future": {f: [[o, g, b] for (ff, o, g, b), _ in sorted(((k, v) for k, v in SC[sp][it["id"]].items() if k[0] == f), key=lambda kv: -kv[1])]
                          for f in it["gold"]["future"]}} for it in its]
    agg = metrics.aggregate_sga([metrics.sga_scores(it["gold"], r) for it, r in zip(its, ranked)])
    d = [a["f1"] - b["f1"] for a, b in zip(m, rule)]; rng = random.Random(0)
    bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(4000))
    out[sp] = {"n": len(its), "transition_f1_macro": round(100 * st.mean(x["f1"] for x in m), 1),
               "table_rule_transition_f1_macro": round(100 * st.mean(x["f1"] for x in rule), 1),
               "gbm_minus_table_rule": [round(100 * st.mean(d), 2), round(100 * bs[100], 2), round(100 * bs[3899], 2)],
               **{k: round(100 * v, 1) for k, v in agg.items()}}
    print(sp, out[sp], flush=True)
Path(A_.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(A_.out, "w"), indent=1); print("wrote", A_.out)
