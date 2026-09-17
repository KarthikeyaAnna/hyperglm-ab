#!/usr/bin/env python3
"""HyperGLM-R pilot-13, condition [A] (DESIGN.md) — one Kaggle session on the RTX PRO 6000 (STRICT mode), Qwen3-8B + LoRA.
[B] = reasoning chain + answer, [A] = answer only; the two kernels differ only in CONDITION and train on the same items.

Key design decisions, each aimed at a measured failure mode:
  * slot-timeline prompt + changes-only answer (no more cut-off answers; 'no change' is the default)
  * S=8 sampled answers per item -> vote fractions -> decision threshold chosen on VAL (F1 wants p(change) < 0.5)
    and a confidence-ranked no-constraint list for R@K
  * much more SFT data: SGA13 at three cut points (F 0.9/0.7/0.5) + RR-hard, seeded subset shared by [A] and [B]
Order (each phase skipped if its output exists; target < 5 h):
  1 SFT (5,000 SGA13 + 3,000 RR-hard, 1 epoch) · 2 VAL: SGA 4-sample votes, RR greedy · 3 TEST: same
  4 score VAL (choose tau) · 5 score TEST with tau from VAL
STRICT mode: competition_sources ["arc-prize-2026-arc-agi-3"], enable_internet false, machine_shape NvidiaRtxPro6000;
inputs under /kaggle/input/{datasets,models}; trl & co. from wheels shipped in the dataset.
Knobs (env): N_SGA, N_RR, SFT_HOURS, SAMPLES, BUDGET_H, DRY_RUN (print commands only), KAGGLE_INPUT/KAGGLE_WORKING (tests)."""
import os, sys, json, time, subprocess, shutil, glob
from pathlib import Path

T0 = time.time()
CONDITION = "A"
N_SGA = int(os.environ.get("N_SGA", "5000")); N_RR = int(os.environ.get("N_RR", "3000"))
SFT_HOURS = float(os.environ.get("SFT_HOURS", "2.2")); SAMPLES = int(os.environ.get("SAMPLES", "4"))
BUDGET_H = float(os.environ.get("BUDGET_H", "11.4")); DRY = os.environ.get("DRY_RUN") == "1"   # Kaggle hard stop 12 h; plan targets < 5 h
MODEL_HINT = os.environ.get("MODEL_HINT", "8b")
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input")); WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))

def log(*a): print(f"[{(time.time()-T0)/3600:5.2f}h]", *a, flush=True)
def left(): return BUDGET_H - (time.time() - T0) / 3600

# ---------------------------------------------------------------- locate inputs by marker files (never walk competition data)
def find(root, pattern):
    return sorted(p for p in root.rglob(pattern) if "competitions" not in p.parts) if root.exists() else []
DS_ROOT = INPUT / "datasets" if (INPUT / "datasets").exists() else INPUT
for dp, dns, fns in os.walk(DS_ROOT):
    depth = len(Path(dp).relative_to(DS_ROOT).parts)
    if depth <= 3: print("  " * depth + f"{Path(dp).name}/  ({len(fns)} files: {sorted(fns)[:6]})", flush=True)
    if depth >= 3: dns[:] = []
def one(pattern, what):
    hits = find(DS_ROOT, pattern); assert hits, f"{what}: no {pattern} under {DS_ROOT}"; return hits[0]
SCRIPTS = one("generate_r13.py", "scripts").parent
SRC = one("r13.py", "src").parents[1]
TASKS_SRC = one("ag_sga13_val.jsonl", "tasks").parent
WHEELS_SRC = one("trl-*.whl", "wheels").parent
CODE = WORK / "hyperglm_r"; CODE.mkdir(parents=True, exist_ok=True)
for src_dir, name in ((SCRIPTS, "scripts"), (SRC, "src"), (TASKS_SRC, "tasks"), (WHEELS_SRC, "wheels")):
    if not (CODE / name).exists(): shutil.copytree(src_dir, CODE / name, ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
TASKS = CODE / "tasks"; WHEELS = CODE / "wheels"
MODEL_ROOT = INPUT / "models" if (INPUT / "models").exists() else INPUT
safetensors = [p for p in find(MODEL_ROOT, "*.safetensors") if MODEL_HINT in str(p).lower()] or find(MODEL_ROOT, "*.safetensors")
assert safetensors, "no attached model (model_sources) found"
MODEL = str(safetensors[0].parent)
RUNS = WORK / "runs" / f"p13{CONDITION}"; RUNS.mkdir(parents=True, exist_ok=True)
log("scripts", SCRIPTS, "| src", SRC, "| tasks", TASKS_SRC, "| wheels", WHEELS_SRC, "| model", MODEL)

# ---------------------------------------------------------------- offline deps
PY = sys.executable
if not DRY:
    def pip_offline(*pkgs):
        r = subprocess.run([PY, "-m", "pip", "install", "--no-index", f"--find-links={WHEELS}", *pkgs], capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True); return r.returncode == 0
    def can_import(mod):
        return subprocess.run([PY, "-c", f"import {mod}; print({mod}.__version__)"], capture_output=True, text=True).returncode == 0
    if not can_import("trl"):
        log("trl missing -> offline install"); ok = pip_offline("trl")
        if not ok or not can_import("trl"):
            log("installing the whole stack from wheels"); pip_offline("transformers", "peft", "accelerate", "datasets", "huggingface_hub", "tokenizers", "safetensors", "trl")
    subprocess.run([PY, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True, text=True)   # Kaggle's torchao 0.10 breaks peft
    chk = subprocess.run([PY, "-c", "import trl, transformers, peft, torch; print('trl', trl.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)"], capture_output=True, text=True)
    print(chk.stdout, chk.stderr[-2000:], flush=True); assert chk.returncode == 0, "dependency stack not importable"
    import torch; log("gpu", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
    assert "RTX PRO 6000" in torch.cuda.get_device_name(0) or os.environ.get("ALLOW_ANY_GPU"), "not on the RTX PRO 6000 — check competition_sources/enable_internet/machine_shape"


def sh(cmd, name):
    log("RUN", name); print(" ".join(map(str, cmd)), flush=True)
    if DRY: return True
    r = subprocess.run(list(map(str, cmd)), cwd=str(CODE), env=dict(os.environ, HYPERGLM_LOG_MEM="1")); log("END", name, "rc", r.returncode)
    return r.returncode == 0


SGA_TRAIN, RR_TRAIN = TASKS / "ag_sga13_train.jsonl", TASKS / "ag_rr_train.jsonl"
FILES = {sp: {"sga": TASKS / f"ag_sga13_{sp}.jsonl", "rr": TASKS / f"ag_rr_{sp}.jsonl"} for sp in ("val", "test")}
NEW_TOKENS = {("B", "sga"): 1800, ("A", "sga"): 800, ("B", "rr"): 200, ("A", "rr"): 40}     # longest gold answer on test: 1424 / 570 tokens
N_ITEMS = {sp: {t: sum(1 for _ in open(f)) for t, f in d.items()} for sp, d in FILES.items()}
log("condition", CONDITION, "| items", N_ITEMS, "| train SGA", N_SGA, "RR", N_RR, "| samples", SAMPLES)


def sft(cond):
    out = RUNS / f"sft_{cond}"
    if (out / "adapter_config.json").exists(): return out
    base = [PY, "scripts/train_sft.py", "--model", MODEL, "--train", SGA_TRAIN, RR_TRAIN, "--limits", N_SGA, N_RR, "--select_seed", 0,
            "--max_target_chars", 4000, "--condition", cond, "--out", out, "--max_length", 4096, "--epochs", 1, "--lr", 2e-4,
            "--group_by_length", "--save_steps", 100, "--resume"]
    # effective batch 16 either way; length grouping runs the longest batch first, so an OOM at micro-batch 8 costs ~1 minute
    ok = sh(base + ["--batch", 8, "--grad_accum", 2, "--max_hours", SFT_HOURS], f"sft {cond} (SGA13 {N_SGA} + RR {N_RR}, micro-batch 8, length-grouped)")
    if not ok and not DRY:
        hours = round(SFT_HOURS - (time.time() - T0) / 3600, 2)
        ok = sh(base + ["--batch", 4, "--grad_accum", 4, "--max_hours", max(0.5, hours)], f"sft {cond} retry at micro-batch 4")
    return out if (ok and (out / "adapter_config.json").exists()) or DRY else None


def gen(cond, adapter, split, task, samples):
    out = RUNS / f"gen_{cond}_{split}_{'greedy' if samples == 1 else f'vote{samples}'}_{task}.json"
    if out.exists() and not DRY and json.load(open(out))["meta"]["n"] >= N_ITEMS[split][task]: return out
    hours = left() - 0.1
    if hours < 0.1: log("skip", out.name, "budget"); return None
    cmd = [PY, "scripts/generate_r13.py", "--model", MODEL, "--adapter", adapter, "--files", FILES[split][task], "--condition", cond,
           "--out", out, "--max_new_tokens", NEW_TOKENS[(cond, task)], "--max_hours", round(hours, 2)]
    cmd += ["--greedy", "--seqs_per_batch", 128] if samples == 1 else ["--samples", samples, "--temperature", 1.0, "--seqs_per_batch", 128]
    sh(cmd, out.name)
    return out if out.exists() or DRY else None


def complete(p, split, task):
    return DRY or (p is not None and json.load(open(p))["meta"]["n"] >= N_ITEMS[split][task])


def score(split, paths, tau_from=None):
    out = RUNS / f"score_{CONDITION}_{split}.json"
    if not all(complete(p, split, t) for p, t in zip(paths, ("sga", "rr"))):
        log("not scoring", out.name, "(generation incomplete — re-run the kernel to resume)"); return None
    cmd = [PY, "scripts/score_r13.py", "--gen", *paths, "--split", split, "--tasks_dir", TASKS, "--out", out]
    if tau_from: cmd += ["--tau_from", tau_from]
    sh(cmd, out.name); return out


adapter = sft(CONDITION)
if adapter:
    gens = {sp: [gen(CONDITION, adapter, sp, "sga", SAMPLES), gen(CONDITION, adapter, sp, "rr", 1)] for sp in ("val", "test")}
    s_val = score("val", gens["val"])
    if s_val: score("test", gens["test"], tau_from=s_val)

summary = {p.stem: json.load(open(p)) for p in sorted(RUNS.glob("score_*.json"))}
json.dump(summary, open(RUNS / "summary.json", "w"), indent=1)
log("SUMMARY")
for k, v in summary.items():
    s, r = v.get("sga", {}), v.get("rr", {})
    print(f"{k:22s} SGA trans F1 {s.get('model', {}).get('transition_f1_macro')} (table rule {s.get('table_rule_0.75', {}).get('transition_f1_macro')}) "
          f"R@10 {s.get('model', {}).get('R@10')} (persist {s.get('persistence', {}).get('R@10')}, ranked table {s.get('ranked_table', {}).get('R@10')}) "
          f"| RR acc {r.get('model', {}).get('accuracy')} (object prior {r.get('object_prior', {}).get('accuracy')}, prior|groups {r.get('prior_given_other_groups', {}).get('accuracy')})", flush=True)
if not DRY:
    for ck in glob.glob(str(RUNS / "*/checkpoint-*")): shutil.rmtree(ck, ignore_errors=True)
    for d in ("tasks", "wheels"): shutil.rmtree(CODE / d, ignore_errors=True)   # keep the output small
log("DONE")
