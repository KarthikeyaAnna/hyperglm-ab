# Pilot-13 results (17 Sep 2026) — scored once on TEST, thresholds from VAL

Kernels: `<kaggle-user>/hyperglm-r-pilot-13-{b,a}` (RTX PRO 6000, Qwen3-8B + LoRA).
Parse rate 99.9 % / 97.9 %, no answer cut off.
τ chosen on val: [B] 0.50, [A] 0.25. Raw generations: `runs/p13B_v2/`, `runs/p13A_v2/`.

## TEST (1,785 SGA / 1,810 RR-hard items)

| | transition F1 macro | micro | R@10 | mR@10 | RR-hard acc |
|---|---|---|---|---|---|
| **[B] reasoning** | **26.3** | 26.6 | **88.6** | 62.0 | **65.4** |
| [A] answer only | 22.8 | 26.4 | 89.3 | 65.3 | 65.1 |
| persistence / object prior | 17.8 | 0.0 | 75.5 | 57.5 | 62.5 |
| table rule (θ 0.75) | 30.2 | 30.3 | 65.7 | 41.1 | — |
| ranked table | — | — | 89.9 | 64.6 | — |
| prior given other groups | — | — | — | — | 69.9 |
| oracle | 100 | 100 | 96.7 | 87.3 | — |
| non-LLM reference (gradient boosting) | 40.7 | — | 88.1 | 63.1 | — |

## Pre-registered endpoints (DESIGN.md), [B] on TEST

| # | endpoint | result | verdict |
|---|---|---|---|
| E1 | transition F1 vs table rule | −3.92 [−5.62, −2.25] | **FAIL** |
| E2 | R@10 vs persistence | +13.15 [+12.35, +14.01] | **PASS** |
| E3 | R@10 vs ranked table | −1.30 [−1.54, −1.07] | **FAIL** |
| E4 | RR-hard vs object prior | +2.87 [+1.44, +4.31] | **PASS** |
| E5 | RR-hard vs prior given other groups | −4.48 [−5.97, −2.98] | **FAIL** |

Secondary, paired on the same TEST items:
* **[B] − [A] transition F1 +3.44 [+1.92, +5.10]** — the reasoning chain helps (consistent with val result +3.37 [+1.13, +5.58]).
* [B] − [A] RR-hard accuracy +0.33 [−0.72, +1.38] — no difference.

## What the fixes did and did not do

Fixed: no cut-off answers (parse 99.9 %); R@10 75.5 → 88.6 via the ranked vote list;
val-chosen τ replaces greedy's implicit 0.5 cut-off (for [B] it *chose* 0.5, so the gain came from elsewhere).

Still losing on changes. [A] predicts 13.4 new predicates per item and [B] 5.25, against ~3 gold introductions:
both over-predict changes, and the table rule's precision (30.3 micro) is still better. The non-LLM reference at 40.7
shows ~14 points of graph-only signal the LLM is not using: dwell time, horizon, slot history, other groups.
RR-hard: both conditions land at ~65, between the object prior (62.5) and the conditional prior (69.9); the chain
adds nothing there, so the models are not exploiting the other two groups of the same object.

## Next (not started)

1. Put the features the reference model uses into the prompt/chain (dwell, horizon, per-slot history summary,
   the other groups' current state) — the LLM currently has to infer them from the timeline.
2. RR-hard: give the conditional statistic in the prompt, or train only on the slots where it is informative.
3. Only then consider GRPO on transition F1 (verifiable reward), starting from the [B] adapter.
