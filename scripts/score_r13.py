#!/usr/bin/env python3
"""Score pilot-13 generations (DESIGN.md). CPU only.

SGA (ag_sga13_*): vote fractions over the S answers of each item ->
  * decision set at threshold tau -> transition F1 (macro / micro)          tau: chosen on VAL, reused on TEST
  * confidence-ranked list (votes, ties by table confidence) -> R@K / mR@K (no constraint)
  baselines on the same items: persistence, table rule (stay < 0.75, chosen on val), ranked table, oracle
RR-hard (ag_rr_*): majority vote over parsed answers -> accuracy, macro-F1
  baselines: object prior, prior given the object's other two groups (both from train frames only)
Paired bootstrap (item level, 4000 resamples, seed 0) against the strongest baseline of each metric.

  python scripts/score_r13.py --gen runs/p13/gen_B_val_votes.json --split val --out runs/p13/score_B_val_votes.json
  python scripts/score_r13.py --gen runs/p13/gen_B_test_votes.json --split test --tau_from runs/p13/score_B_val_votes.json ..."""
import sys, json, argparse, random, statistics as st, collections, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
DATA = Path(os.environ.get("HYPERGLM_DATA", ROOT / "data"))
from hyperglm_r import metrics, procedural, r13
from hyperglm_r.data import GROUPS

ap = argparse.ArgumentParser()
ap.add_argument("--gen", nargs="+", required=True); ap.add_argument("--split", choices=["val", "test"], required=True)
ap.add_argument("--tasks_dir", default=str(DATA / "tasks_r13")); ap.add_argument("--tau", type=float)
ap.add_argument("--tau_from", help="score JSON from VAL whose chosen tau is reused"); ap.add_argument("--out", required=True)
a = ap.parse_args()
T = Path(a.tasks_dir); P = procedural.load(T / "procedural_ag.json"); PRI = json.load(open(T / "rr_priors.json"))
assert not (a.split == "test" and a.tau is None and a.tau_from is None), "on TEST the threshold must come from VAL (--tau_from)"
tau_fixed = a.tau if a.tau is not None else (json.load(open(a.tau_from))["sga"]["tau"] if a.tau_from else None)

gold = {}
for name in (f"ag_sga13_{a.split}.jsonl", f"ag_rr_{a.split}.jsonl"):
    for l in open(T / name):
        it = json.loads(l); gold[it["id"]] = it

gens, meta = {}, {}
for g in a.gen:
    d = json.load(open(g)); meta[g] = d["meta"]
    for r in d["items"]: gens[r["id"]] = r["outputs"]
sga = [gold[i] for i in gens if gold.get(i, {}).get("task") == "sga13"]
rr = [gold[i] for i in gens if gold.get(i, {}).get("task") == "rr"]
missing = [i for i in gens if i not in gold]
assert not missing, f"{len(missing)} generated ids not in {T} ({a.split}): {missing[:3]}"


def boot(d, n=4000):
    rng = random.Random(0); m = st.mean(d)
    bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return {"delta": round(100 * m, 2), "ci95": [round(100 * bs[int(.025 * n)], 2), round(100 * bs[int(.975 * n) - 1], 2)], "n": len(d)}


def micro(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return 100 * (2 * p * r / (p + r) if p + r else 0.0)


def rk(items, preds):
    agg = metrics.aggregate_sga([metrics.sga_scores(it["gold"], p) for it, p in zip(items, preds)])
    return {k: round(100 * v, 1) for k, v in agg.items()}


res = {"split": a.split, "gen": meta}
if sga:
    S = max(len(gens[it["id"]]) for it in sga)
    V = {}; parsed = []
    for it in sga:
        V[it["id"]], n = r13.votes(it, gens[it["id"]]); parsed.append(n / len(gens[it["id"]]))
    none_parsed = {it["id"] for it in sga if not V[it["id"]] and not any(r13.parse_changes(t)[0] for t in gens[it["id"]])}

    def trans(it, tau):
        if it["id"] in none_parsed: return metrics.transition_scores(it["gold"], None, it["observed"])
        return metrics.transition_scores(it["gold"], r13.decide(it, V[it["id"]], tau), it["observed"])

    grid = sorted({round(k / S, 4) for k in range(1, S + 1)} | ({0.5} if S == 1 else set()))
    grid_scores = {t: 100 * st.mean(trans(it, t)["f1"] for it in sga) for t in grid}
    tau = tau_fixed if tau_fixed is not None else max(grid, key=lambda t: (grid_scores[t], -abs(t - 0.5)))
    model_tr = [trans(it, tau) for it in sga]
    persist = [metrics.persist_prediction(it["gold"], it["observed"]) for it in sga]
    rule = [metrics.table_rule_prediction(it["gold"], it["observed"], P, 0.75) for it in sga]
    pers_tr = [metrics.transition_scores(it["gold"], p, it["observed"]) for it, p in zip(sga, persist)]
    rule_tr = [metrics.transition_scores(it["gold"], p, it["observed"]) for it, p in zip(sga, rule)]
    model_rank = [r13.ranked(it, V[it["id"]], P) if it["id"] not in none_parsed else {"future": {}} for it in sga]
    table_rank = [r13.ranked(it, {}, P, use_votes=False) for it in sga]
    oracle = [{"future": {f: [[o, g, p] for o, gs in c.items() for g in GROUPS for p in gs[g]] for f, c in it["gold"]["future"].items()}} for it in sga]
    per_r10 = lambda preds: [metrics.sga_scores(it["gold"], p)["R@10"] for it, p in zip(sga, preds)]
    res["sga"] = {
        "n": len(sga), "samples": S, "parse_rate": round(100 * st.mean(parsed), 1), "items_without_any_parse": len(none_parsed),
        "tau": tau, "tau_source": "fixed" if tau_fixed is not None else f"chosen on {a.split}", "tau_grid_transition_f1": {str(k): round(v, 2) for k, v in grid_scores.items()},
        "model": {"transition_f1_macro": round(100 * st.mean(x["f1"] for x in model_tr), 1), "transition_f1_micro": round(micro(model_tr), 1), **rk(sga, model_rank)},
        "persistence": {"transition_f1_macro": round(100 * st.mean(x["f1"] for x in pers_tr), 1), "transition_f1_micro": round(micro(pers_tr), 1), **rk(sga, persist)},
        "table_rule_0.75": {"transition_f1_macro": round(100 * st.mean(x["f1"] for x in rule_tr), 1), "transition_f1_micro": round(micro(rule_tr), 1), **rk(sga, rule)},
        "ranked_table": rk(sga, table_rank), "oracle": rk(sga, oracle),
        "bootstrap": {
            "transition_f1: model - table_rule": boot([m["f1"] - r["f1"] for m, r in zip(model_tr, rule_tr)]),
            "transition_f1: model - persistence": boot([m["f1"] - r["f1"] for m, r in zip(model_tr, pers_tr)]),
            "R@10: model - ranked_table": boot([m - r for m, r in zip(per_r10(model_rank), per_r10(table_rank))]),
            "R@10: model - persistence": boot([m - r for m, r in zip(per_r10(model_rank), per_r10(persist))]),
        },
    }
if rr:
    def prior(it, conditional):
        obj, g, f = it["meta"]["object"], it["meta"]["group"], str(it["meta"]["frame"])
        base = PRI["object_group"].get(f"{obj}/{g}")
        if not conditional: return base
        fp = collections.defaultdict(set)
        for o, g2, p in it["observed"][f]:
            if o == obj and g2 != g: fp[g2].add(p)
        return PRI["object_group_others"].get(f"{obj}/{g}/" + "|".join(" ".join(sorted(fp[g2])) for g2 in GROUPS if g2 != g), base)

    def majority(texts):
        c = collections.Counter(); order = {}
        for t in texts:
            p = metrics.parse(t)
            if p["ok"] and isinstance(p["answer"].get("predicates"), list):
                k = tuple(sorted(set(map(str, p["answer"]["predicates"])))); c[k] += 1; order.setdefault(k, len(order))
        if not c: return None
        return list(max(c, key=lambda k: (c[k], -order[k])))

    rows = {"model": [], "object_prior": [], "prior_given_other_groups": []}; parsed = []
    for it in rr:
        m = majority(gens[it["id"]]); parsed.append(m is not None)
        rows["model"].append(metrics.rr_score(it["gold"], {"predicates": m} if m is not None else {}))
        for name, cond in (("object_prior", False), ("prior_given_other_groups", True)):
            pr = prior(it, cond); rows[name].append(metrics.rr_score(it["gold"], {"predicates": pr.split(" ") if pr else []}))
    res["rr"] = {"n": len(rr), "parse_rate": round(100 * st.mean(parsed), 1),
                 **{k: {kk: round(100 * vv, 1) for kk, vv in metrics.aggregate_rr(v).items()} for k, v in rows.items()},
                 "bootstrap": {f"accuracy: model - {b}": boot([m["exact"] - r["exact"] for m, r in zip(rows["model"], rows[b])]) for b in ("object_prior", "prior_given_other_groups")}}

Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
if sga:
    s = res["sga"]
    print(f"SGA {a.split}: n={s['n']} samples={s['samples']} parse {s['parse_rate']}% tau={s['tau']} ({s['tau_source']})")
    print(f"  {'':18s} {'trans F1 macro':>14s} {'micro':>6s} {'R@10':>6s} {'mR@10':>6s} {'R@20':>6s} {'mR@20':>6s}")
    for k in ("model", "persistence", "table_rule_0.75", "ranked_table", "oracle"):
        r = s[k]; print(f"  {k:18s} {str(r.get('transition_f1_macro', '')):>14s} {str(r.get('transition_f1_micro', '')):>6s} {r['R@10']:6.1f} {r['mR@10']:6.1f} {r['R@20']:6.1f} {r['mR@20']:6.1f}")
    for k, v in s["bootstrap"].items(): print(f"  {k:36s} {v['delta']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
if rr:
    q = res["rr"]
    print(f"RR-hard {a.split}: n={q['n']} parse {q['parse_rate']}% | model acc {q['model']['accuracy']} mF1 {q['model']['macro_f1']} | object prior {q['object_prior']['accuracy']} | prior|other groups {q['prior_given_other_groups']['accuracy']}")
    for k, v in q["bootstrap"].items(): print(f"  {k:44s} {v['delta']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
print("wrote", a.out)
