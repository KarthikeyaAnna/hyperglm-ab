"""Pilot-13 SGA format (DESIGN.md): slot-timeline serialisation, changes-only answers, vote-based decoding.

Why:
* full-graph answers + long chains were cut off at the token limit ~9% of the time (scored 0);
* a single decision set cannot compete on R@K, which ranks a no-constraint list (ranked table reaches 92 R@10);
* greedy decoding only predicts a change when p(change) > 0.5, while F1 on rare changes wants a lower threshold.

Prompt  = the observed graph grouped per (object, group) slot as a run-length timeline
          (a temporal hyperedge per slot), plus the transition table for current states and the future frame gaps.
Answer  = {"changes": [[object, group, [predicates], frame], ...]}; every slot not listed keeps its current state,
          a change holds from its frame until the slot's next change. Lossless w.r.t. the gold future.
Chain   = one <infer> line per slot with input-derived evidence (current state, dwell, stay prob, top next, previous
          state) and the decision. Only the decision comes from the gold future.
Decoding for metrics: S sampled answers -> per-(frame, object, group, predicate) vote fraction ->
          decision set at a threshold chosen on VAL (transition F1) and a confidence-ranked list (R@K).
"""
import json, re, collections
from .data import GROUPS, frame_preds
from .procedural import successors
from .tasks import SYSTEM, _observed_index

SCHEMA13 = '{"changes": [["<object>", "<group>", ["<predicate>", ...], <frame>], ...]}'
RULES_B = ("Reason first inside <infer>...</infer>: one line per object/group slot listed above, giving its current state, "
           "how many frames it has held, its stay probability, its most likely next state, then '-> stays' or "
           "'-> <new state> from <frame>'. Then give the final JSON inside <answer>...</answer>.")
MAX_SEGMENTS = 8


def _state_txt(s):
    return " ".join(s) if s else "-"


def slot_timeline(seq, frames, obj, g):
    """run-length segments [(state_tuple|None, first_frame, last_frame, n_frames)] over observed frames; None = absent."""
    segs = []
    for fp, fr in zip(seq, frames):
        st = tuple(sorted(fp[obj][g])) if obj in fp else None
        if segs and segs[-1][0] == st:
            s, f0, _, n = segs[-1]; segs[-1] = (s, f0, fr["frame_index"], n + 1)
        else:
            segs.append((st, fr["frame_index"], fr["frame_index"], 1))
    return segs


def slot_evidence(P, seq, frames, obj, g):
    segs = slot_timeline(seq, frames, obj, g)
    cur = segs[-1][0] or (); dwell = segs[-1][3]
    prev = next((s for s, *_ in reversed(segs[:-1]) if s is not None and s != cur), None)
    if cur:
        # the least stable current predicate drives the decision
        rows = [(successors(P, g, obj, p, 3), p) for p in cur]
        (st, nxt), _ = min(rows, key=lambda r: r[0][0])
    else:
        st, nxt = None, []
    return {"segments": segs, "current": cur, "dwell": dwell, "previous": prev, "stay": st, "next": nxt}


def serialize13(obs, P, objs, future_frames):
    seq = [frame_preds(fr) for fr in obs]
    last = obs[-1]["frame_index"]
    scene = sorted({o["category"] for fr in obs for o in fr["objects"] if o["category"] != "person"})
    lines = [f"objects seen: person, {', '.join(scene)}",
             f"observed annotated frames: {len(obs)} (frame {obs[0]['frame_index']} to {last})",
             "slot timelines (object/group: state@first-last frame | ...; '-' = none, 'absent' = object not annotated):"]
    ev = {}
    for obj in objs:
        for g in GROUPS:
            e = slot_evidence(P, seq, obs, obj, g); ev[(obj, g)] = e
            segs = e["segments"]; hidden = len(segs) - MAX_SEGMENTS
            parts = [("absent" if s is None else _state_txt(s)) + (f"@{f0}" if f0 == f1 else f"@{f0}-{f1}") for s, f0, f1, _ in segs[-MAX_SEGMENTS:]]
            lines.append(f"  {obj}/{g}: " + (f"... {hidden} earlier | " if hidden > 0 else "") + " | ".join(parts) + f" (now, {e['dwell']} frames)")
    lines.append("transitions for current states (predicate -> prob it stays | next predicate and its prob given a change):")
    for obj in objs:
        for g in GROUPS:
            for p in ev[(obj, g)]["current"]:
                st, nxt = successors(P, g, obj, p, 3)
                lines.append(f"  {obj}/{g}: {p} -> stay {st:.2f}" + ("".join(f" | {b} {pr:.2f}" for b, pr in nxt)))
    lines.append("future frames to predict (frame (+frames after the last observed)): " + ", ".join(f"{f} (+{f - last})" for f in future_frames))
    return "\n".join(lines), ev


def gold_changes(last_preds, objs, fut):
    """chronological set-changes per slot relative to the running state; frames where the object is absent are skipped."""
    changes = []
    state = {(obj, g): tuple(sorted(last_preds[obj][g])) for obj in objs for g in GROUPS}
    for fr in fut:
        fp = frame_preds(fr)
        for obj in objs:
            if obj not in fp: continue
            for g in GROUPS:
                s = tuple(sorted(fp[obj][g]))
                if s != state[(obj, g)]:
                    changes.append([obj, g, list(s), fr["frame_index"]]); state[(obj, g)] = s
    return changes


def sga13_items(videos, P, split, F=0.9):
    for vid, frames in videos.items():
        N = len(frames)
        if N < 3: continue
        n_obs = max(1, min(N - 1, int(round(F * N))))
        obs, fut = frames[:n_obs], frames[n_obs:]
        last = frame_preds(obs[-1]); objs = list(last)
        if not objs: continue
        gold = {}
        for fr in fut:
            fp = frame_preds(fr)
            cell = {obj: {g: fp[obj][g] for g in GROUPS} for obj in objs if obj in fp}
            if cell: gold[str(fr["frame_index"])] = cell
        if not gold: continue
        future_frames = [fr["frame_index"] for fr in fut]
        h_text, ev = serialize13(obs, P, objs, future_frames)
        changes = gold_changes(last, objs, fut)
        by_slot = collections.defaultdict(list)
        for obj, g, s, f in changes: by_slot[(obj, g)].append((s, f))
        infer = []
        for obj in objs:
            for g in GROUPS:
                e = ev[(obj, g)]
                stay = f", stay {e['stay']:.2f}" if e["stay"] is not None else ""
                top = f", top next {e['next'][0][0]} {e['next'][0][1]:.2f}" if e["next"] else ""
                before = f", before {_state_txt(e['previous'])}" if e["previous"] is not None else ""
                out = ", then ".join(f"{_state_txt(s)} from {f}" for s, f in by_slot.get((obj, g), [])) or "stays"
                infer.append(f"{obj}/{g}: {_state_txt(e['current'])} for {e['dwell']} frames{stay}{top}{before} -> {out}")
        instruction = (f"Observed {n_obs} of {N} annotated frames. For the objects present in the last observed frame "
                       f"({', '.join(objs)}), predict every change of their attention / spatial / contacting predicates in the "
                       f"future frames. A slot keeps its current state unless a change is listed; a change holds from its frame "
                       f"until that slot's next change. Schema: {SCHEMA13}")
        ans = json.dumps({"changes": changes}, separators=(",", ":"))
        user_A = h_text + "\n\n" + instruction + "\nRespond with the JSON only."
        user_B = h_text + "\n\n" + instruction + "\n\n" + RULES_B
        yield {"id": f"sga13-{F:.1f}:{split}:{vid}", "task": "sga13", "video": vid, "split": split, "F": F,
               "n_change": len(changes),
               "prompt_A": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_A}],
               "prompt_B": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_B}],
               "target_A": ans,
               "target_B": "<infer>\n" + "\n".join(infer) + "\n</infer>\n<answer>" + ans + "</answer>",
               "gold": {"future": gold}, "observed": _observed_index(obs),
               "meta": {"F": F, "n_obs": n_obs, "n_future": len(fut), "objects": objs, "future_frames": future_frames}}


# ------------------------------------------------------------------------------------------- parsing / decoding
_ANS = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_changes(text):
    """-> (ok, changes, n_bad_entries). ok requires a JSON object with a 'changes' list; malformed entries are dropped."""
    m = _ANS.search(text)
    body = m.group(1) if m else text
    cands = [body]
    if "{" in body: cands.append(body[body.find("{"): body.rfind("}") + 1])
    for c in cands:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if not (isinstance(obj, dict) and isinstance(obj.get("changes"), list)): continue
        good, bad = [], 0
        for e in obj["changes"]:
            try:
                o, g, preds, f = e
                if isinstance(preds, str): preds = [preds]
                if g not in GROUPS or not isinstance(o, str) or not all(isinstance(p, str) for p in preds): raise ValueError
                good.append([o, g, sorted(set(preds)), int(f)])
            except Exception:
                bad += 1
        return True, good, bad
    return False, [], 0


def apply_changes(gold, observed, changes):
    """changes -> full future prediction on the gold layout (frames x objects), like persist_prediction."""
    from .metrics import _last_observed
    last = _last_observed(observed)
    by_slot = collections.defaultdict(list)
    for o, g, preds, f in changes: by_slot[(o, g)].append((int(f), list(preds)))
    for v in by_slot.values(): v.sort(key=lambda x: x[0])
    out = {"future": {}}
    for f, cell in gold.get("future", {}).items():
        fi = int(f); out["future"][f] = {}
        for obj in cell:
            out["future"][f][obj] = {}
            for g in GROUPS:
                state = sorted(last.get((obj, g), set()))
                for cf, preds in by_slot.get((obj, g), []):
                    if cf <= fi: state = preds
                out["future"][f][obj][g] = state
    return out


def table_conf(P, obj, g, lastset):
    """graph-only confidence for candidates of a slot: current predicate 0.5+0.5*stay, successor (1-stay)*prob."""
    c = {}
    for a in lastset:
        st, nxt = successors(P, g, obj, a, 5)
        c[a] = max(c.get(a, 0.0), 0.5 + 0.5 * st)
        for b, pr in nxt:
            if b not in lastset: c[b] = max(c.get(b, 0.0), (1 - st) * pr)
    return c


def votes(it, texts):
    """-> (vote fraction per (frame, obj, group, pred), n_parsed). Unparsed samples abstain."""
    cnt = collections.Counter(); n = 0
    for t in texts:
        ok, ch, _ = parse_changes(t)
        if not ok: continue
        n += 1
        dec = apply_changes(it["gold"], it["observed"], ch)
        for f, cell in dec["future"].items():
            for obj, gs in cell.items():
                for g in GROUPS:
                    for p in gs[g]: cnt[(f, obj, g, p)] += 1
    return ({k: v / n for k, v in cnt.items()} if n else {}), n


def decide(it, vote, tau, keep=0.5):
    """decision set: current predicates kept with vote >= keep, introduced predicates added with vote >= tau."""
    from .metrics import _last_observed
    last = _last_observed(it["observed"]); out = {"future": {}}
    for f, cell in it["gold"]["future"].items():
        out["future"][f] = {}
        for obj in cell:
            out["future"][f][obj] = {}
            for g in GROUPS:
                l = last.get((obj, g), set()); s = set()
                for (vf, vo, vg, p), v in vote.items():
                    if vf == f and vo == obj and vg == g and v >= (keep if p in l else tau): s.add(p)
                out["future"][f][obj][g] = sorted(s)
    return out


def ranked(it, vote, P, use_votes=True):
    """confidence-ranked no-constraint list per frame: score = vote fraction, ties broken by the table confidence.
    use_votes=False gives the ranked-table baseline (current predicates vote 1, everything else 0)."""
    from .metrics import _last_observed
    last = _last_observed(it["observed"]); out = {"future": {}}
    for f, cell in it["gold"]["future"].items():
        cand = {}
        for obj in cell:
            for g in GROUPS:
                l = last.get((obj, g), set())
                for p, c in table_conf(P, obj, g, l).items():
                    v = vote.get((f, obj, g, p), 0.0) if use_votes else float(p in l)
                    cand[(obj, g, p)] = (v, c)
                if use_votes:
                    for (vf, vo, vg, p), v in vote.items():
                        if vf == f and vo == obj and vg == g and (obj, g, p) not in cand: cand[(obj, g, p)] = (v, 0.0)
        out["future"][f] = [list(k) for k, _ in sorted(cand.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))]
    return out
