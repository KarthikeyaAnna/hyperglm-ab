#!/usr/bin/env python3
"""Pilot-13 generation (DESIGN.md): raw outputs only, scoring is done by score_r13.py (CPU, re-runnable offline).

--samples 1 --greedy : one greedy answer per item
--samples S          : S sampled answers per item (temperature --temperature, no top-k/top-p truncation)
Items are batched by prompt length; progress is flushed to --out every few batches and resumed on restart."""
import sys, json, argparse, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import torch
from hyperglm_r.common import read_items, load_model_and_tok, chat_kwargs

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--adapter"); ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--condition", choices=["A", "B"], required=True); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int); ap.add_argument("--samples", type=int, default=1); ap.add_argument("--greedy", action="store_true")
ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--max_new_tokens", type=int, default=1200)
ap.add_argument("--seqs_per_batch", type=int, default=96, help="prompts per batch = max(1, seqs_per_batch // samples)")
ap.add_argument("--max_hours", type=float, default=None); ap.add_argument("--attn", default=None); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
assert not (a.greedy and a.samples != 1), "--greedy means one sample"

T0 = time.time(); torch.manual_seed(a.seed)
items = read_items(a.files, a.limit)
out_p = Path(a.out); out_p.parent.mkdir(parents=True, exist_ok=True)
done = {}
if out_p.exists():
    prev = json.load(open(out_p)); done = {r["id"]: r for r in prev["items"]}
    print(f"[gen] resuming: {len(done)} items already generated", flush=True)
todo = [it for it in items if it["id"] not in done]
model, tok, dt = load_model_and_tok(a.model, adapter=a.adapter, attn=a.attn); model.eval()
if a.adapter: model = model.merge_and_unload()     # merged LoRA: same outputs, no per-layer adapter overhead while decoding
ck = chat_kwargs(tok)
texts = {it["id"]: tok.apply_chat_template(it[f"prompt_{a.condition}"], tokenize=False, add_generation_prompt=True, **ck) for it in todo}
lens = {k: len(tok(v)["input_ids"]) for k, v in texts.items()}
todo.sort(key=lambda it: -lens[it["id"]])          # longest first: an OOM shows up immediately


def meta():
    return {"model": a.model, "adapter": a.adapter, "condition": a.condition, "samples": a.samples, "greedy": a.greedy,
            "temperature": None if a.greedy else a.temperature, "max_new_tokens": a.max_new_tokens, "files": a.files}


def flush():
    json.dump({"meta": {**meta(), "seconds": round(time.time() - T0), "n": len(done)}, "items": list(done.values())}, open(out_p, "w"))


per_batch = max(1, a.seqs_per_batch // a.samples); i = 0; nb = 0; n_trunc = 0
while i < len(todo):
    if a.max_hours and time.time() - T0 > a.max_hours * 3600:
        print(f"[gen] time limit reached with {len(todo) - i} items left", flush=True); break
    batch = todo[i:i + per_batch]
    enc = tok([texts[it["id"]] for it in batch], return_tensors="pt", padding=True).to(model.device)
    kw = dict(max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id, num_return_sequences=a.samples)
    kw.update(dict(do_sample=False, temperature=None, top_p=None, top_k=None) if a.greedy else dict(do_sample=True, temperature=a.temperature, top_p=1.0, top_k=0))
    try:
        with torch.no_grad():
            gen = model.generate(**enc, **kw)
    except torch.cuda.OutOfMemoryError:
        del enc; torch.cuda.empty_cache()
        assert per_batch > 1, "OOM with a single prompt"
        per_batch = max(1, per_batch // 2); print(f"[gen] OOM -> {per_batch} prompts per batch", flush=True); continue
    L = enc["input_ids"].shape[1]
    for j, it in enumerate(batch):
        outs = []
        for s in range(a.samples):
            g = gen[j * a.samples + s][L:]
            n_trunc += int(tok.eos_token_id not in g.tolist() and tok.pad_token_id not in g.tolist())
            outs.append(tok.decode(g, skip_special_tokens=True))
        done[it["id"]] = {"id": it["id"], "task": it["task"], "outputs": outs}
    i += len(batch); nb += 1
    if nb % 5 == 0 or i >= len(todo):
        flush(); print(f"[gen] {len(done)}/{len(items)} items | {time.time() - T0:.0f}s | no-EOS so far {n_trunc}", flush=True)
flush()
print(f"[gen] wrote {out_p}: {len(done)}/{len(items)} items, {round(time.time() - T0)}s, sequences without EOS {n_trunc}", flush=True)
if torch.cuda.is_available():
    print(f"[mem] peak allocated {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
