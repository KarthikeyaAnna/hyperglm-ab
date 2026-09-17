#!/usr/bin/env python3
"""SFT for condition [A] (answer only) or [B] (chain + answer). LoRA via peft, TRL SFTTrainer,
completion-only loss on a prompt/completion conversational dataset. Resumable."""
import sys, json, argparse
import torch, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from hyperglm_r.common import read_items, load_model_and_tok, chat_kwargs

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--train", nargs="+", required=True)
ap.add_argument("--condition", choices=["A", "B"], required=True); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int); ap.add_argument("--tasks", nargs="*")
ap.add_argument("--max_steps", type=int, default=-1); ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lora_r", type=int, default=64)
ap.add_argument("--max_length", type=int, default=4096); ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--grad_accum", type=int, default=16); ap.add_argument("--save_steps", type=int, default=200)
ap.add_argument("--resume", action="store_true"); ap.add_argument("--attn", default=None)
ap.add_argument("--max_target_chars", type=int, default=None, help="keep items whose CHAIN (target_B) fits; applied before --limit, same item set for [A] and [B]")
ap.add_argument("--limits", type=int, nargs="+", default=None, help="per-file item caps, one per --train file (pilot-13); overrides --limit")
ap.add_argument("--select_seed", type=int, default=None, help="with --limits: random subset (seeded) instead of the file head; same subset for [A] and [B]")
ap.add_argument("--max_hours", type=float, default=None, help="stop training cleanly after this many hours and save the adapter")
ap.add_argument("--group_by_length", action="store_true", help="batch items of similar length together (less padding); the longest batch runs first so an OOM shows up at step 1")
a = ap.parse_args()

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

if a.limits:
    import random
    assert len(a.limits) == len(a.train), "--limits needs one cap per --train file"
    items = []
    for _f, _cap in zip(a.train, a.limits):
        _its = read_items([_f], None, a.tasks)
        if a.max_target_chars: _its = [it for it in _its if len(it["target_B"]) <= a.max_target_chars]
        if _cap < len(_its):
            _its = random.Random(a.select_seed).sample(_its, _cap) if a.select_seed is not None else _its[:_cap]
        print(f"[sft] {_f}: using {len(_its)} items", flush=True); items += _its
elif a.max_target_chars:
    items = []
    for _f in a.train:
        _its = [it for it in read_items([_f], None, a.tasks) if len(it["target_B"]) <= a.max_target_chars]
        items += _its[: a.limit] if a.limit else _its
    print(f"[sft] length filter target_B <= {a.max_target_chars} chars -> {len(items)} items", flush=True)
else:
    items = read_items(a.train, a.limit, a.tasks)
model, tok, dt = load_model_and_tok(a.model, attn=a.attn)
ck = chat_kwargs(tok)
# Qwen3-style templates: eval/GRPO run with enable_thinking=False, which puts an empty think block
# right after the generation prompt. SFTConfig cannot pass that kwarg, so we put the same block at
# the start of the completion -> identical token sequence at train and inference time.
prefix = "<think>\n\n</think>\n\n" if ck else ""
ds = Dataset.from_list([{"prompt": it[f"prompt_{a.condition}"],
                         "completion": [{"role": "assistant", "content": prefix + it[f"target_{a.condition}"]}]} for it in items])
print(f"[sft] condition {a.condition} | {len(ds)} examples | model {a.model} | think-prefix {bool(prefix)}")
def _length_grouping(on):
    """transformers 5.0 (Kaggle image) spells it group_by_length; 5.2 replaced it with train_sampling_strategy"""
    fields = SFTConfig.__dataclass_fields__
    if "train_sampling_strategy" in fields: return {"train_sampling_strategy": "group_by_length" if on else "random"}
    if "group_by_length" in fields: return {"group_by_length": on}
    assert not on, "this transformers version has no length-grouped sampler"
    return {}


cfg = SFTConfig(output_dir=a.out, max_length=a.max_length, completion_only_loss=True, packing=False,
                per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
                learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
                num_train_epochs=a.epochs, max_steps=a.max_steps, logging_steps=5, save_steps=a.save_steps,
                save_total_limit=2, bf16=(str(dt) == "torch.bfloat16"), fp16=(str(dt) == "torch.float16"),
                gradient_checkpointing=True, report_to="none", seed=0,
                **_length_grouping(a.group_by_length))
peft_cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
callbacks = []
if a.max_hours:
    import time as _time
    from transformers import TrainerCallback
    class TimeLimit(TrainerCallback):
        """stop cleanly at a wall-clock budget so the adapter is always saved before the Kaggle session ends"""
        t0 = _time.time()
        def on_step_end(self, args, state, control, **kw):
            if _time.time() - self.t0 > a.max_hours * 3600:
                print(f"[sft] time limit {a.max_hours}h reached at step {state.global_step}/{state.max_steps}", flush=True)
                control.should_training_stop = True
            return control
    callbacks.append(TimeLimit())
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg, callbacks=callbacks)
last = None
if a.resume and Path(a.out).exists():
    cks = sorted(Path(a.out).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    last = str(cks[-1]) if cks else None
trainer.train(resume_from_checkpoint=last)
trainer.save_model(a.out)          # saves the adapter
json.dump({"condition": a.condition, "model": a.model, "n": len(ds), "args": vars(a)}, open(Path(a.out) / "sft_meta.json", "w"), indent=1)
print("[sft] saved adapter to", a.out)

import os as _os
if _os.environ.get("HYPERGLM_LOG_MEM") and torch.cuda.is_available():
    print(f"[mem] peak allocated {torch.cuda.max_memory_allocated()/1e9:.1f} GB | reserved {torch.cuda.max_memory_reserved()/1e9:.1f} GB", flush=True)
