"""Parsing + metrics for the [A]/[B] experiment: SGA R@K / mR@K (no-constraint, ranked or set output),
transition F1 (are the predicted CHANGES right), RR accuracy / macro-F1, and the non-learned baselines."""
import re, json, collections
from .data import GROUPS

TAG = {t: re.compile(rf"<{t}>(.*?)</{t}>", re.S) for t in ("cite", "infer", "answer")}


def parse(text):
    """-> dict(ok, answer, cites:list[str], has_tags)."""
    out = {"ok": False, "answer": None, "cites": [], "has_tags": False}
    m = TAG["answer"].search(text)
    body = m.group(1) if m else text
    out["has_tags"] = bool(m)
    c = TAG["cite"].search(text)
    if c: out["cites"] = [l.strip() for l in c.group(1).splitlines() if l.strip()]
    # find a JSON object in body
    for cand in (body, body[body.find("{"): body.rfind("}") + 1] if "{" in body else ""):
        try:
            ans = json.loads(cand)
            # a bare JSON object is not enough: it must carry the task's top-level key with the right type
            if isinstance(ans, dict) and (isinstance(ans.get("future"), dict) or isinstance(ans.get("predicates"), list)):
                out["answer"] = ans; out["ok"] = True; break
            out["answer"] = ans
        except Exception: continue
    return out


def _flatten_future(future):
    """-> {frame: ordered list of (obj, group, pred)} preserving emission order (=rank)."""
    res = {}
    if not isinstance(future, dict): return res
    for f, cell in future.items():
        lst, seen = [], set()
        if isinstance(cell, list):          # confidence-ranked no-constraint list [[obj, group, pred], ...] (pilot-13)
            for t in cell:
                if isinstance(t, (list, tuple)) and len(t) == 3 and t[1] in GROUPS:
                    k = (str(t[0]), t[1], str(t[2]))
                    if k not in seen: seen.add(k); lst.append(k)
            res[str(f)] = lst
            continue
        if not isinstance(cell, dict): continue
        for obj, gs in cell.items():
            if not isinstance(gs, dict): continue
            for g in GROUPS:
                for p in (gs.get(g) or []) if isinstance(gs.get(g), list) else []:
                    k = (str(obj), g, str(p))
                    if k not in seen: seen.add(k); lst.append(k)
        res[str(f)] = lst
    return res


def sga_scores(gold, pred, ks=(10, 20, 50)):
    """Per-frame recall@K averaged over future frames; mR@K = mean over predicate classes.
    Returns dict R@k, mR@k plus per-class hit/total for aggregation."""
    G = _flatten_future(gold.get("future", {})); Pd = _flatten_future((pred or {}).get("future", {}) if isinstance(pred, dict) else {})
    out = {}; cls = {k: collections.defaultdict(lambda: [0, 0]) for k in ks}
    for k in ks:
        rs = []
        for f, gt in G.items():
            gts = set(gt); top = set(Pd.get(f, [])[:k])
            if not gts: continue
            rs.append(len(gts & top) / len(gts))
            for (o, g, p) in gts:
                cls[k][p][1] += 1; cls[k][p][0] += (o, g, p) in top
        out[f"R@{k}"] = sum(rs) / len(rs) if rs else 0.0
    out["_cls"] = {k: {p: v for p, v in d.items()} for k, d in cls.items()}
    return out


def aggregate_sga(per_item, ks=(10, 20, 50)):
    agg = {}
    for k in ks:
        agg[f"R@{k}"] = sum(x[f"R@{k}"] for x in per_item) / max(1, len(per_item))
        tot = collections.defaultdict(lambda: [0, 0])
        for x in per_item:
            for p, (h, t) in x["_cls"][k].items(): tot[p][0] += h; tot[p][1] += t
        agg[f"mR@{k}"] = sum(h / t for h, t in tot.values() if t) / max(1, len(tot))
    return agg


def rr_score(gold, pred):
    g = set(map(str, gold.get("predicates", []))); p = set(map(str, (pred or {}).get("predicates", []) if isinstance(pred, dict) and isinstance(pred.get("predicates"), list) else []))
    return {"exact": float(g == p), "tp": len(g & p), "fp": len(p - g), "fn": len(g - p),
            "per_pred": {x: (x in p, x in g) for x in g | p}}


def aggregate_rr(per_item):
    acc = sum(x["exact"] for x in per_item) / max(1, len(per_item))
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    for x in per_item:
        for pr, (inp, ing) in x["per_pred"].items():
            if inp and ing: tp[pr] += 1
            elif inp: fp[pr] += 1
            else: fn[pr] += 1
    f1s = []
    for pr in set(tp) | set(fp) | set(fn):
        P_ = tp[pr] / max(1, tp[pr] + fp[pr]); R_ = tp[pr] / max(1, tp[pr] + fn[pr])
        f1s.append(2 * P_ * R_ / max(1e-9, P_ + R_))
    return {"accuracy": acc, "macro_f1": sum(f1s) / max(1, len(f1s))}


def _last_observed(observed):
    """-> {(obj, group): set(preds)} in the last observed frame."""
    last_f = max((f for f in observed if not f.startswith("_")), key=int)
    d = collections.defaultdict(set)
    for obj, g, p in observed[last_f]: d[(obj, g)].add(p)
    return d


def persist_prediction(gold, observed):
    last = _last_observed(observed)
    return {"future": {f: {obj: {g: sorted(last.get((obj, g), set())) for g in GROUPS} for obj in cell} for f, cell in gold.get("future", {}).items()}}


# ----------------------------------------------------------------------------- fair transition scoring
def transition_scores(gold, pred, observed):
    """Score the predicates an answer INTRODUCES (not in the slot's last observed set) against the gold
    introductions, over gold slots only. Objects absent from a gold future frame are ignored; a slot the answer
    omits counts as kept; pred=None (unparsed) misses every gold introduction and scores f1=0.
    Unlike sga_change_scores, a false change on an UNCHANGED slot counts as a false positive."""
    last = _last_observed(observed)
    ok = isinstance(pred, dict) and isinstance(pred.get("future"), dict)
    fut = pred["future"] if ok else {}
    tp = fp = fn = 0
    for f, cell in gold.get("future", {}).items():
        pcell = fut.get(f) if isinstance(fut.get(f), dict) else {}
        for obj, gs in cell.items():
            pgs = pcell.get(obj) if isinstance(pcell.get(obj), dict) else None
            for g in GROUPS:
                l = last.get((obj, g), set()); gi = set(gs.get(g) or []) - l
                if not ok: fn += len(gi); continue
                pl = pgs.get(g) if (pgs is not None and isinstance(pgs.get(g), list)) else sorted(l)
                pi = set(pl) - l
                tp += len(gi & pi); fp += len(pi - gi); fn += len(gi - pi)
    if not ok: return {"tp": 0, "fp": 0, "fn": fn, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def table_rule_prediction(gold, observed, P, thr=0.75):
    """Non-learned baseline that reads only the prompt's transition table: move each live predicate to its top-1
    successor when its stay probability < thr, else keep it. thr=0.75 was chosen on VAL. Uses the gold
    future layout (frames/objects) exactly like persist_prediction."""
    from .procedural import successors
    last = _last_observed(observed); out = {"future": {}}
    for f, cell in gold.get("future", {}).items():
        out["future"][f] = {}
        for obj in cell:
            out["future"][f][obj] = {}
            for g in GROUPS:
                new = set()
                for a in last.get((obj, g), set()):
                    stay, nx = successors(P, g, obj, a, 1)
                    new.add(nx[0][0] if (nx and stay < thr) else a)
                out["future"][f][obj][g] = sorted(new)
    return out
