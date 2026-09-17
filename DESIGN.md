# Pilot-13: beat the simple baselines on held-out data (pre-registered 17 Sep 2026)

Written **before** any pilot-13 output exists. Thresholds are chosen on VAL only; TEST is scored once.

## Why baseline generation lost to "copy the last frame"

| failure | evidence | fix in pilot-13 |
|---|---|---|
| answers cut off at the token limit → scored 0 | 86–91 / 1,000 outputs have no `</answer>`; median output 2,500 chars (the `<cite>` block restates the graph) | answer lists **changes only** (median 48 tokens); chain is one short line per slot (median 279 tokens) |
| one decision set cannot compete on R@K | on finished answers R@10 70–77 vs persistence 77; a *ranked* no-constraint list of the table reaches 92.4 on val | model output → **vote fractions** over 4 samples → confidence-ranked list (ties broken by table confidence) |
| greedy decoding predicts a change only when p(change) > 0.5 | SFT ties the table on changes (29.9 vs 29.9 on finished answers); F1 on rare changes wants a lower cut-off (table rule uses stay < 0.75) | decision threshold τ on the vote fraction, **chosen on VAL** |
| chains stated the outcome with no input evidence | `<infer>` said "stays" / "-> X from f" only | every `<infer>` line carries input-derived evidence: current state, dwell, stay prob, top next, previous state |
| little training data | 500 SFT items per file | 5,000 SGA items drawn from three cut points (F 0.9/0.7/0.5) + 3,000 RR-hard, seeded subset shared by [A] and [B] |
| process reward rewarded "no change" | 59.5 % empty answers, R@10 up, transition F1 down | no GRPO in pilot-13 |

## Headroom check (CPU, before any GPU time; val, fit on train minus val)

| predictor | transition F1 macro | R@10 | mR@10 | RR-hard acc |
|---|---|---|---|---|
| persistence / object prior | 19.6 | 75.0 | 63.7 | 59.6 |
| table rule (stay < 0.75) | 32.2 | 66.3 | 47.6 | — |
| ranked table (persist + successors) | — | 92.8 | 75.3 | — |
| gradient boosting on graph features (τ tuned on the other half of val) | 38.7 / 38.8 | 91.9 | 75.2 | — |
| prior given the object's other two groups | — | — | — | 66.9 |

So the graph carries about +6 transition-F1 points beyond the first-order table, and RR-hard about +7 beyond the object prior.
`scripts/gbm_reference_r13.py` reports the gradient-boosting reference on TEST with thresholds from VAL (`results/gbm_reference.json`).

## Data (`scripts/build_tasks_r13.py` → `data/tasks_r13/`)

Same videos, split and procedural graph (train 6,951 / val 826 / test 1,810 videos; graph from train minus val).
Build-time assertions: split disjointness, lossless changes ↔ future round trip for every item, no future frame in any timeline.

## Protocol

* Two separate Kaggle kernels, `kaggle/run_pilot13_B.py` and `kaggle/run_pilot13_A.py` (identical except CONDITION), each planned to finish in < 5 h.
* Model: Qwen3-8B + LoRA (r 64, α 128, all linear), SFT 1 epoch on 5,000 SGA13 + 3,000 RR-hard (time guard 2.2 h), lr 2e-4 cosine, eff. batch 16, max length 4,096; chains over 4,000 chars excluded.
* Generation: SGA **4 samples at temperature 1.0** (no top-k / top-p); RR-hard greedy. Adapter merged for decoding.
* SGA scoring (`scripts/score_r13.py`): unparsed samples abstain; an item with no parsed sample scores 0.
  Decision set = current predicates with vote ≥ 0.5 plus introduced predicates with vote ≥ τ. τ ∈ {1/4, 1/2, 3/4, 1}, chosen on VAL
  by macro transition F1, reused on TEST. R@K uses the ranked list.
* RR-hard: the greedy answer.
* Baselines on the same items: persistence, table rule (θ 0.75, chosen on val), ranked table, oracle,
  object prior, prior given the object's other two groups (train frames only).
* Paired bootstrap, item level, 4,000 resamples, seed 0.

## Pre-registered endpoints (TEST, [B] with 4-sample votes)

| # | endpoint | "beats the baseline" means |
|---|---|---|
| E1 | macro transition F1 vs **table rule** | paired 95 % CI lower bound > 0 |
| E2 | R@10 vs **persistence** | CI lower bound > 0 |
| E3 | R@10 vs **ranked table** | CI lower bound > 0 (the fair R@K comparison) |
| E4 | RR-hard accuracy vs **object prior** | CI lower bound > 0 |
| E5 | RR-hard accuracy vs **prior given other groups** | CI lower bound > 0 |

Secondary: [B] vs [A] (does the chain help?) on E1/E4, paired on the same items; LLM vs gradient-boosting reference (reported, not a gate).
Dropped to fit < 5 h per kernel (17 Sep, before any output): 8 samples, greedy SGA passes, zero-shot Qwen3-8B.

Reference (non-LLM, test, thresholds from val): gradient boosting 40.7 transition F1 vs table rule 30.2 (+10.5 [+8.9, +12.2]); R@10 88.1, mR@10 63.1.
A failed endpoint is reported as failed. No threshold, prompt or checkpoint is changed after looking at TEST.
