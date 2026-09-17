"""Build reasoning items from canonical graphs.

SGA  (Scene Graph Anticipation, SceneSayer protocol): observe the first F of a video's annotated
     frames, predict every object's predicates (3 groups) in ALL remaining frames, for objects
     present in the last observed frame.
RR   (Relation Reasoning, VSGR's definition on clean labels): one (frame, object, group) slot is
     masked as '?'; infer its predicates from the rest of the frame and the recent history.

Every item carries: prompts for condition [A] (answer only) and [B] (chain + answer), targets,
gold answer, the gold SUPPORT set (what a correct chain must cite), and the OBSERVED graph
(what a chain may cite) so rewards/metrics can verify groundedness without re-reading data.
"""
import json, random
from .data import GROUPS, frame_preds
from .procedural import successors
from .serialize import serialize

SYSTEM = ("You are given a video scene graph: per annotated frame, each object's relationships to the "
          "person in three groups (attention / spatial / contacting), plus transition statistics. "
          "Answer strictly in the requested format.")

CHAIN_RULES = ("Reason first inside <cite>...</cite> and <infer>...</infer>, then give the final JSON inside "
               "<answer>...</answer>. Each <cite> line must be one of: "
               "'<object>/<group>: <predicate> @<frame>' copied from an observed frame, or "
               "'<object>/<group>: <predicate> -> <next> <prob>' copied from the transition table. "
               "Cite only what appears in the graph above.")

SGA_SCHEMA = '{"future": {"<frame>": {"<object>": {"attention": [...], "spatial": [...], "contacting": [...]}}}}'
RR_SCHEMA = '{"predicates": [...]}'


def _cite_obs(obj, g, p, f): return f"{obj}/{g}: {p} @{f}"
def _cite_tr(obj, g, p, b, pr): return f"{obj}/{g}: {p} -> {b} {pr:.2f}"


def _observed_index(frames):
    return {str(fr["frame_index"]): sorted([obj, g, p] for obj, gs in frame_preds(fr).items() for g in GROUPS for p in gs[g])
            for fr in frames}


def _messages(h_text, instruction, chain):
    user = h_text + "\n\n" + instruction + ("\n\n" + CHAIN_RULES if chain else "\nRespond with the JSON only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ----------------------------------------------------------------------------------------- SGA
def sga_items(videos, P, split, F=0.9, k_succ=3):
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
        support, cites, infer = [], [], []
        for obj in objs:
            for g in GROUPS:
                for p in last[obj][g]:
                    support.append([obj, g, p, obs[-1]["frame_index"]])
                    cites.append(_cite_obs(obj, g, p, obs[-1]["frame_index"]))
                    st, nxt = successors(P, g, obj, p, k_succ)
                    for b, pr in nxt: cites.append(_cite_tr(obj, g, p, b, pr))
                    # templated inference that explains the gold
                    # collapse the per-frame outcome into one clause: 'stays' or '-> X from frame f'
                    change = next(((f, cell[obj][g]) for f, cell in gold.items() if obj in cell and p not in cell[obj][g]), None)
                    outcome = "stays" if change is None else f"-> {' '.join(change[1]) or '-'} from {change[0]}"
                    top = f"; top next {nxt[0][0]} {nxt[0][1]:.2f}" if nxt else ""  # decision evidence next to the decision
                    infer.append(f"{obj}/{g}: {p} (stay {st:.2f}{top}) {outcome}")
        h_text = serialize(obs, P, k=k_succ)
        instruction = (f"Observed {n_obs} of {N} annotated frames. Predict, for each of the {len(fut)} future frames "
                       f"{[fr['frame_index'] for fr in fut]}, the predicates of each object present in the last frame "
                       f"({', '.join(objs)}). Schema: {SGA_SCHEMA}")
        ans = json.dumps({"future": gold}, separators=(",", ":"))
        n_change = sum(1 for f, cell in gold.items() for obj, gs in cell.items() for g in GROUPS if set(gs[g]) != set(last[obj][g]))
        yield {"id": f"sga{F:.1f}:{split}:{vid}", "task": "sga", "video": vid, "split": split, "F": F, "n_change": n_change,
               "prompt_A": _messages(h_text, instruction, False), "prompt_B": _messages(h_text, instruction, True),
               "target_A": ans,
               "target_B": "<cite>\n" + "\n".join(cites) + "\n</cite>\n<infer>\n" + "\n".join(infer) + "\n</infer>\n<answer>" + ans + "</answer>",
               "gold": {"future": gold}, "support": support, "observed": _observed_index(obs),
               "meta": {"F": F, "n_obs": n_obs, "n_future": len(fut), "objects": objs}}


# ----------------------------------------------------------------------------------------- RR
def rr_items(videos, P, split, per_video=2, context=3, seed=0, group_weights=(0.15, 0.25, 0.6), hard=True):
    """hard=True: the masked slot is hidden in EVERY context frame (no history to copy) -> infer from the
    object's other groups, the scene, and class priors. hard=False: history visible (VSGR-style partial info).
    In both, the transition table excludes the masked slot."""
    rng = random.Random(seed)
    for vid, frames in videos.items():
        cands = []
        for ti, fr in enumerate(frames):
            for obj, gs in frame_preds(fr).items():
                for g in GROUPS:
                    if gs[g]: cands.append((ti, obj, g))
        if not cands: continue
        w = dict(zip(GROUPS, group_weights))
        picks = set()
        for _ in range(per_video * 6):
            if len(picks) >= per_video: break
            c = rng.choices(cands, weights=[w[g] for _, _, g in cands])[0]; picks.add(c)
        for ti, obj, g in sorted(picks):
            fr = frames[ti]; ctx = frames[max(0, ti - context): ti + 1]
            fp = frame_preds(fr); gold = fp[obj][g]
            support, cites, infer = [], [], []
            for g2 in GROUPS:
                if g2 == g: continue
                for p in fp[obj][g2]:
                    support.append([obj, g2, p, fr["frame_index"]]); cites.append(_cite_obs(obj, g2, p, fr["frame_index"]))
            if ti > 0 and not hard:
                pp = frame_preds(frames[ti - 1]).get(obj)
                if pp:
                    for p in pp[g]:
                        support.append([obj, g, p, frames[ti - 1]["frame_index"]]); cites.append(_cite_obs(obj, g, p, frames[ti - 1]["frame_index"]))
                        st, nxt = successors(P, g, obj, p, 3)
                        for b, pr in nxt: cites.append(_cite_tr(obj, g, p, b, pr))
                        infer.append(f"{obj}/{g} was {p} in frame {frames[ti-1]['frame_index']} (stay {st:.2f}); " +
                                     ("it persists" if p in gold else f"it changes to {' '.join(gold)}"))
            others = " ".join(p for g2 in GROUPS if g2 != g for p in fp[obj][g2])
            infer.append(f"in frame {fr['frame_index']} {obj} is {others or 'unannotated'} in the other groups, consistent with {g} = {' '.join(gold)}")
            h_text = serialize(ctx, P, mask=(obj, g), mask_frame=None if hard else fr["frame_index"], mask_all=hard)
            instruction = (f"The '?' marks the missing {g} predicate(s) of '{obj}' in frame {fr['frame_index']}"
                           + (" (hidden in every frame)." if hard else ".")
                           + f" Infer them from the rest of the graph{'' if hard else ' and the history'}. Schema: {RR_SCHEMA}")
            ans = json.dumps({"predicates": gold}, separators=(",", ":"))
            yield {"id": f"{'rr' if hard else 'rreasy'}:{split}:{vid}:{fr['frame_index']}:{obj}:{g}", "task": "rr", "video": vid, "split": split, "variant": "hard" if hard else "easy",
                   "prompt_A": _messages(h_text, instruction, False), "prompt_B": _messages(h_text, instruction, True),
                   "target_A": ans,
                   "target_B": "<cite>\n" + "\n".join(cites) + "\n</cite>\n<infer>\n" + "\n".join(infer) + "\n</infer>\n<answer>" + ans + "</answer>",
                   "gold": {"predicates": gold}, "support": support,
                   "observed": {**_observed_index(ctx), "_masked": [obj, g, str(fr["frame_index"])]},
                   "meta": {"frame": fr["frame_index"], "object": obj, "group": g, "context": len(ctx), "hard": hard}}
