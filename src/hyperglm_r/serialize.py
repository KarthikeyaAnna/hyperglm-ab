"""H_text: the hypergraph as text. Object-centric TOON-style rows (subject is always the person)
plus the procedural-graph successor table for the last observed frame. Identical for
conditions [A] and [B]; the *only* thing that differs between them is the target."""
from .data import GROUPS, frame_preds
from .procedural import successors


def frame_row(fr, mask=None):
    """'frame 439: dish: looking_at in_front_of wiping | table: ...'. mask=(obj, group) -> '?'."""
    cells = []
    for obj, gs in frame_preds(fr).items():
        parts = []
        for g in GROUPS:
            if mask and mask == (obj, g):
                parts.append("?")
            else:
                parts.append(" ".join(gs[g]) if gs[g] else "-")
        cells.append(f"{obj}: " + " / ".join(parts))
    return f"frame {fr['frame_index']}: " + " | ".join(cells)


def transition_rows(fr, P, k=3, mask=None):
    rows = []
    for obj, gs in frame_preds(fr).items():
        for g in GROUPS:
            if mask and (obj, g) == mask:      # never leak the masked slot through the transition table
                continue
            for p in gs[g]:
                st, nxt = successors(P, g, obj, p, k)
                nx = " | ".join(f"{b} {pr:.2f}" for b, pr in nxt)
                rows.append(f"{obj}/{g}: {p} -> stay {st:.2f}" + (f" | {nx}" if nx else ""))
    return rows


def serialize(frames, P=None, mask=None, mask_frame=None, k=3, mask_all=False):
    """mask=(obj, group) hides that slot: in frame `mask_frame` only, or in every frame if mask_all
    (RR-hard: no history for the masked slot). Transitions never include the masked slot."""
    objs = sorted({o["category"] for fr in frames for o in fr["objects"] if o["category"] != "person"})
    lines = ["objects: person, " + ", ".join(objs),
             "columns per object: attention / spatial / contacting"]
    for fr in frames:
        lines.append(frame_row(fr, mask if (mask_all or mask_frame is None or fr["frame_index"] == mask_frame) else None))
    if P is not None:
        lines.append("transitions from last frame (predicate -> prob it stays | next predicate and its prob given a change):")
        lines += ["  " + r for r in transition_rows(frames[-1], P, k, mask=mask)]
    return "\n".join(lines)
