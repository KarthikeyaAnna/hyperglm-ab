# HyperGLM-R · Does explicit reasoning help an LLM predict video scene graphs?

A single controlled experiment on Action Genome. Two models, identical in every respect except one:

* **[A] answer only** — given the scene graph, the model outputs the answer.
* **[B] reasoning + answer** — the model first writes one short evidence line per relationship, then the answer.

Same base model, same training items, same prompts, same decoding, same test items, same metrics. So whatever [B]
wins by is attributable to the reasoning chain and not to the format, the data, or the decoding.

**Headline result (Action Genome test, 1,785 videos, scored once):** reasoning improves change prediction by
**+3.44 transition F1 [+1.92, +5.10]** over the answer-only control. Neither condition beats the strongest
non-learned baseline on that metric, and this README says exactly where the remaining gap is.

---

## 1. Background: what problem is this?

A **video scene graph** is a per-frame, symbolic description of what is happening: nodes are the person and the
objects, edges are the relationships between them. In Action Genome one frame might be:

```
person --looking_at--> cup        person --in_front_of--> cup      person --holding--> cup
person --not_looking_at--> table  person --beneath--> table        person --not_contacting--> table
```

Two things people do with these graphs:

* **Scene graph generation** — produce the graph from pixels. This is a perception problem, and it is *not* what
  this project studies. We treat the graph as given.
* **Scene graph anticipation / reasoning** — given the graph so far, say what happens next, or fill in what is
  missing. This is a reasoning problem over symbols, and it is what we study.

**HyperGLM** (CVPR 2025, arXiv 2411.18042) feeds scene graphs to an LLM (Mistral-7B) together with a *procedural
graph* — a table of how often each relationship is followed by each other relationship — and random-walk
hyperedges. It reports Action Genome anticipation numbers (R@10 38.8 / mR@10 22.3 in its AGS setting), released no
code, and contains **no reasoning step**: the LLM maps graph to answer in one shot.

**Our question:** if the model is made to reason explicitly — to state the evidence it uses before answering —
does it predict the future graph better? That is the entire point of this folder, and it is why the experiment is
built as a paired [A] vs [B] contrast rather than as a leaderboard entry.

---

## 2. Dataset: Action Genome

Action Genome annotates everyday-activity videos frame by frame with the person's relationships to each object.

| | |
|---|---|
| Object classes | 36 (cup, table, door, phone/camera, clothes, …) |
| Relationship types | 25, in three groups |
| **attention** | looking_at, not_looking_at, unsure |
| **spatial** | in_front_of, behind, on_the_side_of, above, beneath, in |
| **contacting** | holding, touching, carrying, wearing, eating, drinking_from, not_contacting, … |
| Subject | always the person |

Because the subject is fixed, the state of a video reduces to a set of **slots**: one slot per
(object, group). A slot holds one or more relationships at a time — e.g. `cup/contacting = {holding}`, or
`table/spatial = {in_front_of, beneath}`. Slots are the unit of prediction throughout this project.

Annotated frames are **sparse and irregular**: a video may be annotated at frames 10, 45, 102, 147 … so "the next
frame" means the next *annotated* frame, and the gap between them varies. Everything below counts annotated frames.

### Split (by video, no overlap)

| split | videos | role |
|---|---|---|
| train | 6,951 | model training and all statistics |
| val | 826 | every threshold and design decision |
| test | 1,810 | final numbers, scored once |

Test is Action Genome's official test set. Val is carved out of train deterministically, by a hash of the video id
(`int(sha1(video_id),16) % 10000 < 1000`), so it is reproducible anywhere and never overlaps test.

**Everything the prompt contains is built from train minus val.** The transition tables and the priors never see a
val or test label. Splits are asserted disjoint at build time, and the build also verifies that no future frame's
content appears anywhere in a prompt.

---

## 3. The two tasks

### SGA — scene graph anticipation (1,785 test items, one per video)

Show the first 90 % of a video's annotated frames; predict every relationship of every object in **all** remaining
frames. This follows the protocol used by SceneSayer and HyperGLM (observed fraction ℱ = 0.9).

Most slots do not change — which is the central difficulty, and the reason the metrics below are designed the way
they are.

### RR-hard — hidden relationship (1,810 test items)

One slot is hidden in **every** frame shown, and the model must infer it from the rest of the graph: the same
object's other two groups, the other objects in the scene, and class priors. This is our reconstruction of
HyperGLM's "relation reasoning" task. HyperGLM defines it on its own VSGR dataset; we audited that release and
found its relationship labels unusable (the recoverable labels are ≈ 93 % uniform noise), so the task was rebuilt
on Action Genome's human annotations.

"Hard" means the masked slot is hidden in all context frames, so it cannot be copied from history. The transition
table shown in the prompt also excludes the masked slot — an earlier version leaked the answer through that table.

---

## 4. Method

### 4.1 The graph becomes text

Per slot, a run-length **timeline** over the observed frames, then HyperGLM's procedural graph (transition
statistics) for the states that are currently active, then the frames to predict:

```
slot timelines (object/group: state@first-last frame | ...):
  phone/camera/contacting: holding@10-147 | touching@166 | holding@201-269 (now, 5 frames)
  phone/camera/attention:  ... 3 earlier | not_looking_at@244 | looking_at@407-435 (now, 1 frames)
transitions for current states (predicate -> prob it stays | next predicate and its prob given a change):
  phone/camera/contacting: holding -> stay 0.97 | touching 0.82 | not_contacting 0.13
future frames to predict (frame (+frames after the last observed)): 512 (+35), 690 (+213)
```

The timeline is the same information as per-frame rows, grouped per slot; it is compact and makes duration and
history visible in one line.

### 4.2 The answer lists only changes

```json
{"changes": [["phone/camera", "attention", ["looking_at"], 690]]}
```

A slot not listed keeps its current state; a listed change holds from its frame until that slot's next change.
This is **lossless** with respect to the full future graph — the build asserts, for every item, that decoding the
change list reproduces the gold future exactly — and it keeps answers short (median 48 tokens instead of ~700),
which is what stopped answers being cut off at the token limit in earlier runs.

### 4.3 [B]'s reasoning chain

One line per slot: the state, how long it has held, the stay probability, the most likely alternative, then the
decision.

```
<infer>
clothes/contacting: wearing (stay 0.96 at 9 frames held; else holding 0.62) -> stays
phone/camera/attention: not_looking_at (stay 0.67 at 1 frames held; else looking_at 0.65; slot changed 8x) -> looking_at from 690
</infer>
<answer>{"changes":[["phone/camera","attention",["looking_at"],690]]}</answer>
```

Every number in a chain line comes from the prompt or from train-set statistics; only the decision after `->` comes
from the gold future. The chains are generated **by code from the gold labels**, not distilled from a larger LLM.
That is deliberate: a distilled chain would make the result a statement about the teacher model, and nothing would
guarantee the reasoning is derivable from the model's own input.

### 4.4 Training: SFT with LoRA

| | |
|---|---|
| Base model | Qwen3-8B, bf16, thinking mode off (see FAQ) |
| Adapter | LoRA r = 64, α = 128, dropout 0.05, on **all linear layers** (~175 M trainable parameters) |
| Objective | supervised fine-tuning, loss on the completion only (the prompt is masked) |
| Data | 5,000 SGA + 3,000 RR-hard items, the **same seeded subset** for [A] and [B] |
| SGA augmentation | items are drawn from three cut points per video (ℱ = 0.9 / 0.7 / 0.5); val and test use 0.9 only |
| Schedule | 1 epoch, lr 2e-4 cosine, warmup 3 %, effective batch 16 (micro-batch 8 × 2 accumulation), max length 4,096 |
| Efficiency | length-grouped batches (≈ 5× faster than random batching, since items run 500–3,800 tokens), gradient checkpointing, LoRA merged into the weights before decoding |
| Hardware | one Kaggle RTX PRO 6000 (96 GB) per condition. [B] 5.2 h total (0.84 h training), [A] 2.0 h (0.59 h training) |

Only the adapter is trained; the base weights are frozen. [A] and [B] differ **only** in which target they are
trained on (`target_A` or `target_B`) and in the instruction line that asks for a chain.

### 4.5 Decoding: votes instead of one answer

An LLM asked for one answer gives one guess. Instead we sample **4 answers per item at temperature 1.0** and
convert them into a **vote fraction** per candidate relationship — the fraction of samples that predicted it.
The votes are used twice:

* **A decision:** predict a change when its vote ≥ τ. τ is chosen on val by transition F1 and then **reused
  unchanged on test** ([B] τ = 0.5, [A] τ = 0.25). This matters because greedy decoding effectively fixes τ = 0.5,
  while the F1-optimal cut-off for rare events is usually lower.
* **A ranking:** candidates sorted by vote (ties broken by the transition table's confidence) give the ranked list
  that R@K needs (see 5.1).

RR-hard uses one greedy answer, since its output is a single set.

---

## 5. Metrics, and why

### 5.1 R@K / mR@K — the standard SGA metrics

For each future frame, take the model's top-K predicted (object, group, relationship) triplets and measure what
fraction of the gold triplets are in there. R@10 averages this; mR@10 averages per relationship class, so rare
relationships count as much as common ones. This is the "no-constraint" protocol: several relationships per slot
are allowed, and ranking matters.

**Read R@K with care.** Because most relationships persist, simply copying the last observed frame scores
**R@10 = 75.5**. A ranked list that keeps the current state and appends the table's likely successors scores
**89.9** with no model at all. So a high R@10 mostly demonstrates a sensible output format, not reasoning.

### 5.2 Transition F1 — our primary metric

Precision, recall and F1 over the relationships an answer **introduces** — those not present in the slot's last
observed frame — against the relationships the gold future actually introduces.

* Predicting a change that does not happen is a false positive.
* Missing a real change is a false negative.
* Saying nothing changes scores 17.8, not 75: there is nowhere to hide.

This is the metric that tracks the actual prediction problem, which is why it is the primary one. (An earlier
version, "change-F1", only inspected slots whose gold changed, so false changes elsewhere were free. It was
replaced for that reason.)

### 5.3 RR-hard: accuracy and macro-F1

Exact set match on the hidden slot, plus macro-F1 over relationship classes.

### 5.4 Statistics

All comparisons are **paired at the item level** with a bootstrap over the test items (4,000 resamples, seed 0).
A difference counts only if the 95 % interval excludes zero. Thresholds come from val; test is scored once.

---

## 6. Baselines (all on the same test items)

| baseline | what it does | why it is there |
|---|---|---|
| **persistence** | predicts nothing changes | the floor any model must clear; exposes R@K inflation |
| **table rule** | moves a relationship to its most likely successor when its stay probability < 0.75 (θ chosen on val) | uses exactly the statistics the prompt shows, with no learning — the honest bar for change prediction |
| **ranked table** | persistence, then the table's successors, ranked | the fair R@K comparison for a ranked output |
| **oracle** | the gold answer | ceiling (R@10 96.7, not 100, because the top-K cut-off truncates dense frames) |
| **object prior** (RR) | the most common relationship for that object and group in train | the floor for RR-hard |
| **prior given other groups** (RR) | the most common relationship given the object's other two groups | a stronger statistical competitor |
| **gradient-boosting reference** | a small non-LLM model over graph features (duration, horizon, slot history, table probabilities), thresholds from val | shows how much of the task is solvable from the graph at all — the honest ceiling for any graph-only method |

The scorer recomputes every baseline on whatever items it scores, so model and baseline numbers can never drift
apart.

---

## 7. Results

### 7.1 [A] vs [B] — the experiment

| Metric | [A] answer only | [B] reasoning | [B] − [A] |
|---|---|---|---|
| **Transition F1** (are the changes right) | 22.8 | **26.3** | **+3.44 [+1.92, +5.10]** ✅ |
| R@10 | **89.3** | 88.6 | −0.70 [−0.96, −0.45] |
| mR@10 | **65.3** | 62.0 | −3.3 (per-class metric, no item-level pairing) |
| RR-hard accuracy | 65.1 | **65.4** | +0.33 [−0.72, +1.38] (no difference) |
| Parse rate | 97.9 % | **99.9 %** | +2.0 (a rate, not a per-item score) |
| Changes predicted per video (≈ 3 are real) | 13.37 | **5.25** | −8.1 (descriptive) |
| Wall-clock for the whole run | 2.0 h | 5.2 h | +3.2 (descriptive) |

**Reasoning helps, and the mechanism is visible.** [A] fires off 13.4 changes per video against about 3 real ones;
[B] is selective at 5.25. Writing the stay probability and duration before deciding makes the model commit less
often and more accurately. The same effect appeared in an earlier run on val, with a different prompt format
(+3.37 [+1.13, +5.58]), so it is not a one-off.

Reasoning costs 2.6× the decoding time, and R@10/mR@10 dip slightly, because those metrics reward listing more
candidates while the chain makes the model more conservative.

### 7.2 Against baselines

| | Transition F1 | R@10 | RR-hard |
|---|---|---|---|
| Copy last frame / most common answer for the object | 17.8 | 75.5 | 62.5 |
| Transition-table rule (no learning) | **30.2** | 65.7 | — |
| Ranked table (persistence + successors) | — | **89.9** | — |
| Prior given the object's other two relationships | — | — | **69.9** |
| **[A] answer only** | 22.8 | 89.3 | 65.1 |
| **[B] reasoning** | 26.3 | 88.6 | 65.4 |
| Gradient-boosting reference (non-LLM) | 40.7 | 88.1 | — |
| Oracle | 100 | 96.7 | — |

Read honestly:

* ✅ [B] beats the naive floors: **R@10 +13.2** over copy-last-frame, **RR-hard +2.9** over the object prior.
* ❌ [B] does **not** beat the transition-table rule on changes (26.3 vs 30.2), nor the ranked table on R@10
  (88.6 vs 89.9), nor the conditional prior on RR-hard (65.4 vs 69.9).
* ❌ Both conditions are well short of the small non-LLM model (40.7).

### 7.3 Where the remaining gap is

Refitting the reference model on subsets of its features (on val, thresholds tuned on one half and scored on the
other) localises it:

| inputs to the reference model | transition F1 |
|---|---|
| table rule, no learning | 32.2 |
| only what the prompt states plainly (object, candidate, stay and successor probabilities) | 32.8 |
| **+ how long the state has held, and the slot's own history** | **38.5** |
| + horizon (which future frame, how many frames ahead) | 33.7 |
| + the other two groups' current state | 32.2 |

So the missing ~6 points are **duration and slot history**. The timeline shows both, but the model does not turn
them into calibrated decisions. Note also that a rule using duration-conditioned probabilities directly scores
28.7 on val, *below* the plain table rule at 32.2 — the gain needs duration *combined* with candidate history, not
duration alone. The other two groups add nothing, which also explains why the chain does not help RR-hard.

---

## 8. Limitations

1. **Ground-truth graphs, not detector output.** Every method here reads human annotations, so our R@10 is not
   comparable with HyperGLM's 38.8 or SceneSayer's 74.8, which start from detectors or video features. Those
   numbers are context only.
2. **The chains are templated.** They are generated by code from gold labels, which keeps them verifiable and
   distillation-free, but their phrasing is fixed. Whether free-form reasoning would do better is untested.
3. **τ is chosen per condition on val** (0.5 for [B], 0.25 for [A]). This is legitimate tuning on a held-out split,
   but the two conditions do not share one threshold.
4. **No zero-shot baseline.** Untrained Qwen3-8B was never scored on these tasks, so the contribution of the
   fine-tuning itself is not isolated. Roughly 1–2 h of GPU would fix this.
5. **One seed per condition.** The paired intervals capture item-level noise, not training-seed noise.
6. **RR-hard reasoning adds nothing** (+0.33). The chain as written does not exploit the object's other groups,
   which the statistics say is where the RR signal is.

---

## 9. What is in this folder

```
README.md          this file
DESIGN.md          the design and pass/fail targets, written and fixed BEFORE the run
RESULTS.md         the full result write-up, including the endpoint verdicts
requirements.txt   pinned versions (transformers, trl, peft) — matching what the runs used

examples/          sga.txt       a real SGA prompt with target [A] and target [B]
                   rr_hard.txt   a real RR-hard prompt with both targets
src/hyperglm_r/    data.py        load per-frame graphs, group them by video
                   procedural.py  transition statistics (HyperGLM's procedural graph), train-only
                   serialize.py   graph -> text for RR-hard (with the masked slot removed everywhere)
                   tasks.py       RR-hard items; the per-frame SGA format the r13 build verifies against
                   r13.py         the format used here: timelines, changes-only answers, chains,
                                  parsing, vote counting, thresholded decisions, ranking
                   metrics.py     R@K / mR@K, transition F1, RR scoring, persistence and table-rule baselines
                   common.py      model + tokenizer + LoRA loading, item reading
scripts/           build_ag_graphs.py    Action Genome annotation pickles -> per-frame graph JSONL
                   build_tasks.py        split, procedural graph, SGA + RR-hard items
                   build_tasks_r13.py    this experiment's items (asserts lossless, same gold, no leak)
                   train_sft.py          SFT + LoRA for [A] or [B]
                   generate_r13.py       greedy or sampled generation; resumable; merges the adapter
                   score_r13.py          all metrics + all baselines + paired bootstrap (CPU)
                   gbm_reference_r13.py  the non-LLM reference (CPU, ~2.5 min)
kaggle/            run_pilot13_B.py, run_pilot13_A.py — one RTX PRO 6000 session each, end to end
                   kernel-metadata.{A,B}.example.json — fill in your Kaggle user and dataset
results/           score_{A,B}_{val,test}.json   every model number in section 7
                   baselines_test_full.json      the no-model baselines
                   gbm_reference.json            the non-LLM reference
tests/             test_r13.py — 6 unit tests for the answer format, decoding, voting and ranking (no data needed)
```

Sanity anchors the test suite and scorer check: gold answers score 100 transition F1; an empty change list scores
exactly persistence (17.8 test / 19.6 val); unparseable output scores 0.

---

## 10. Reproducing it

```bash
pip install -r requirements.txt
export HYPERGLM_DATA=/path/to/data     # Action Genome annotations in $HYPERGLM_DATA/action_genome/annotations

# 1. data (CPU, a few minutes)
python scripts/build_ag_graphs.py      # annotations -> per-frame graphs
python scripts/build_tasks.py          # split + procedural graph + SGA/RR items
python scripts/build_tasks_r13.py      # this experiment's format, with all integrity assertions
python tests/test_r13.py               # 6/6 should pass

# 2. baselines and the non-LLM reference (CPU, minutes — no GPU needed)
python scripts/gbm_reference_r13.py --out results/gbm_reference.json

# 3. train [B] (GPU). [A] is the same command with --condition A
python scripts/train_sft.py --model Qwen/Qwen3-8B --condition B \
  --train $HYPERGLM_DATA/tasks_r13/ag_sga13_train.jsonl $HYPERGLM_DATA/tasks_r13/ag_rr_train.jsonl \
  --limits 5000 3000 --select_seed 0 --max_target_chars 4000 --max_length 4096 \
  --batch 8 --grad_accum 2 --group_by_length --out runs/sft_B

# 4. generate: 4 samples for SGA, greedy for RR-hard, on val and test
python scripts/generate_r13.py --model Qwen/Qwen3-8B --adapter runs/sft_B --condition B \
  --samples 4 --temperature 1.0 --max_new_tokens 1800 \
  --files $HYPERGLM_DATA/tasks_r13/ag_sga13_val.jsonl --out runs/gen_B_val_vote4_sga.json
python scripts/generate_r13.py --model Qwen/Qwen3-8B --adapter runs/sft_B --condition B \
  --greedy --max_new_tokens 200 \
  --files $HYPERGLM_DATA/tasks_r13/ag_rr_val.jsonl --out runs/gen_B_val_greedy_rr.json
#   ... and the same two commands on the _test files

# 5. score: choose the threshold on val, then reuse it on test
python scripts/score_r13.py --split val  --gen runs/gen_B_val_vote4_sga.json runs/gen_B_val_greedy_rr.json \
  --out runs/score_B_val.json
python scripts/score_r13.py --split test --gen runs/gen_B_test_vote4_sga.json runs/gen_B_test_greedy_rr.json \
  --tau_from runs/score_B_val.json --out runs/score_B_test.json
```

`kaggle/run_pilot13_{B,A}.py` run steps 3–5 unattended in one Kaggle session (strict mode: no internet, offline
wheels, the model attached through `model_sources`), and resume if the session is cut short. For [A], SGA uses
`--max_new_tokens 800` and RR-hard `40`, since its answers are much shorter.

Only training and generation need a GPU. Everything else — data building, all baselines, the reference model, and
all scoring — runs on CPU in minutes, so any number in section 7 can be re-derived from saved generations.

---

## 11. FAQ

**Why is Qwen's thinking mode off?** Our reasoning is the `<infer>` chain, which is visible, checkable and
scoreable. Qwen's built-in thinking is hidden free-form text: with it on, [A] would silently reason too and the
control would be meaningless, and we could not verify what the model used. Training with it on would also require
thinking traces, which only a larger model could supply — that is distillation, and it would change the claim.
Implementation detail: the SFT targets begin with the empty `<think></think>` block so the token sequence is
identical at training and at inference.

**Why is transition F1 so much lower than R@10?** They measure different things. R@10 is dominated by
relationships that persist, which are easy; transition F1 looks only at changes, which are hard and rare. An
oracle scores 100 on both; persistence scores 75.5 on R@10 and 17.8 on transition F1.

**Why 4 samples and a threshold instead of one answer?** One greedy answer commits to a change only when the model
believes it is more likely than not. For rare events, the F1-optimal cut-off is lower, and votes over 4 samples
give a coarse probability that a val-chosen threshold can exploit. It also produces the ranking R@K needs.

**Is the comparison with HyperGLM fair?** No, and we do not make it. We read ground-truth graphs; they start from
detectors or video features. The paper's numbers appear in section 1 as context, never as a claim of improvement.

**What would make the model beat the table rule?** The feature ablation in section 7.3 answers this: give the model
duration and slot-history statistics as numbers rather than expecting it to derive them from the timeline, and only
then consider reinforcement learning against transition F1 directly, starting from the [B] adapter.
