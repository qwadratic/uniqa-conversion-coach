# Plan — Persona distillation v2 (per-step, state-covering, K-sampled)

> **Note.** This is the method writeup for the per-step K-sampled fix. Some scripts it references
> (e.g. `slurm/show_eval.py`, `research/*`) predate the sim_loop refactor and still import the
> gitignored legacy package; the **canonical, current** data path is `sim_loop/to_sft.py` (SFT built
> from the same builders the demo runs). See README → “Distillation”.

Goal: a tiny local persona model whose **emergent S4/S6 churn + conversion match the funnel
anchors** (ε ≤ 0.12, per-persona conv within tol), distilled from the OpenRouter teacher
(`gpt-4o-mini`), runnable fast/offline for the coach loop.

> Background + rationale: `docs/REPORT_distillation_collapse.md`.
> Start over cleanly — new prompt, new dataset, new train/eval. The old `datasets/persona_v1`
> + `sft_steps.jsonl` are superseded (keep for reference).

## Invariants (do not violate)
- **No targets in the prompt.** The 66/24/78 anchors live only in `funnel.py` (eval).
  Conformance comes from the **state sampler + Stage-2 calibration**, never from telling the
  persona its bounce rate.
- **Per-step Markov policy** is the unit: input = (step UI/action-space, running_state,
  disposition, history_brief, [coach_widget]) → output = (events, state_update, decision,
  feeling, thought). Same interface Mode-B coach injection uses.
- **Train on broad states; roll out coherently.** Training examples are independent per-step
  contexts; generation/eval rolls the state forward through the student's own outputs.
- S5/add-on stays out of scope (in-scope flow S1→S2→S3→S4→S6).

---

## Task 1 — Per-step prompt v2  (`build_step_prompt_v2`)
Rebuild the step prompt as a clean, **single-step, Markov** prompt.
- Inputs surfaced: `you_are_on`, `ui_ascii`, `action_space`, `ux_complexity_here`, real
  price block (S4/S6), `your_running_state`, `session_instance` (disposition), `history_brief`,
  optional `coach_widget_shown`.
- Output schema: `events[]`, `state` (attention/satisfaction/effort_left/grasp/effort_vs_reward),
  `feeling`, `decision: continue|leave`, `reason`, `thought`; + `intervention_assessment` when a
  coach widget is shown.
- Rules: decide **honestly** given state vs intent; no mention of any target rate.
- **DoD:** prompt builds for every (persona, step, sampled context); produces valid JSON from
  the teacher on a 20-sample smoke test; contains zero anchor numbers (grep gate).

## Task 2 — State-space sampler  (`sample_state(persona, step, rng)`)
Produce realistic, leave-inclusive contexts.
- Disposition: reuse `_sample_disposition` (persona-weighted; encodes leave-propensity).
- Running state: sample per-step within **plausible** ranges (don't go uniform/off-manifold);
  bias coverage so the leave-prone region (low satisfaction/effort, price-shock dispositions at
  S4/S6) is well represented. Option: seed ranges from a quick "marginal pass" of normal teacher
  sessions, then oversample the tails (DAgger-style).
- **DoD:** sampler yields a balanced spread; a dry-run histogram shows ~1:1 leave/stay in the
  teacher's labels at S4 and S6 (measured on a small probe, Task 3a).

## Task 3 — K-sampled datagen (OpenRouter teacher)
For each (persona, step) draw M contexts; query the teacher **K times** each (temp≈0.9).
- Record per context: the `(messages → output)` SFT pair(s) **and** the empirical
  `leave_rate = #leave/K` (soft target).
- Keep per-K samples as individual SFT rows (natural balancing) OR keep one row + soft-label
  weight — decide in 3a.
- Volume budget (tune): ~M=150 contexts × K=5 × 5 steps × 3 personas ≈ 11k contexts / ~55k
  teacher calls; threaded. Cost check before full run.
- **3a (probe, do first):** M=20, K=5 on S4+S6 for all personas → confirm leave/stay balance
  and that leave-rate tracks disposition. Gate before the full run.
- Writers: `datasets/persona_v2/{sft_steps.jsonl, soft_labels.jsonl, manifest.json}`; commit.
- **DoD:** dataset built at the population mix; per-(persona,step) leave-rate distribution looks
  sane; balanced.

## Task 4 — Train ONE tagged adapter (prompt-internalized)
**Single adapter for all 3 personas**, conditioned on a persona TAG — not 3 adapters, not
the full `persona.md` in the prompt. Enables one-model serving + **cross-persona batching**
(required for population gen, T7b).
- **Student-view** (`research/student_view.py`): teacher labelled with the FULL prompt; student
  trains on `tagged minimal input -> teacher output`, internalizing the static scaffold
  (persona prose, dials, cognitive_model, rules, schema). Measured: **full ~4350 → student ~588
  tok (7.4×, level 1)**; level 2 ~243 (17.9×). One combined `data_tagged/{train,val}.jsonl`.
- Train one adapter on the combined set, both bases (Qwen2.5-1.5B, MiniCPM5-1B). Balance/upweight
  `leave` if needed. Fewer epochs / early-stop on churn-ε.
- **Dials:** v2.1 internalizes dials WITH the tag (fixed per persona); post-hoc dial-tunability
  (dials in prompt) needs dial-AUGMENTED data → deferred v2.2. Outlier fix meanwhile = regenerate
  that persona's cells + retrain (T6).
- **Separability gate:** tag-dropout / swap test — confirm the tag drives behavior (no blending,
  tag not ignored). (old assumption A5.)
- **DoD:** 1 adapter/base saved (`adapter_config.json` present — don't `scancel` mid-save);
  tag-dropout shows distinct per-persona behavior.

## Task 5 — Coherent rollout eval (batched)
- `BatchedLocalTeacher.generate_cohort` (Mode A) → `validate` → per-persona **S4/S6 cond churn +
  conversion + ε** vs `ABANDON_PROBS`. (`slurm/show_eval.py`.)
- Settings that worked: batch=48, N≈60–100, `max_new_tokens≈384`, `expandable_segments`,
  `--time 1:00:00`, robust incremental writes.
- **DoD:** report printed; compare to frontier ε≈0.10 and to v1's ε≈0.48.

## Task 6 — Fix outliers at SOURCE (dials), calibration last-resort
Measure the **coherent-rollout marginal** (T5), NOT the per-step sampled-state rate, then:
1. **Primary — tune dials/disposition** (`research/tune.py`). Structural outliers (e.g. judith
   S4 saturated-high across moods ⇒ disposition-driven, not state) are fixed by shifting that
   persona's dials / `_DISP_W` weights (judith → more "proceed online / prepared for premium")
   so the teacher's behaviour matches anchors → regenerate the affected cells → retrain. This
   fixes the model's *understanding*, and reshapes the training data correctly.
2. **Last-resort — Stage-2 calibration** (per-(persona,step) temperature/threshold on the leave
   decision) ONLY for the small residual after dial-tuning — because dial-tuning costs a full
   regen+retrain loop while calibration is free. Don't lead with it; it rescales a logit, it
   doesn't fix behaviour.
- **DoD:** post-fix rollout ε ≤ 0.12; each persona converts >0; per-persona conv within tol;
  outliers addressed via dials where structural.

> Note: single whole-session prompting is rejected (compare_gen: S4 churn ≡ 0, ε 0.31). It can
> match *overall conversion* (0.11) but not the per-step funnel shape — hence the per-step gate.

## Task 7 — Base comparison + pick
- Qwen2.5-1.5B vs MiniCPM5-1B on {ε, per-persona churn, sessions/sec, calibration effort}.
- **DoD:** one chosen base + a short results table for the report.

## Task 7b — CONTROLLABLE POPULATION generation (headline)
The payoff of the single tagged adapter: generate a synthetic **population** at ANY persona mix
and check the aggregate funnel matches reality.
- `BatchedLocalTeacher.generate_population(weights, N, seed)`: assign each of N sessions a persona
  by `weights` ({judith:.3, franz:.5, peter:.2} — or any mix), roll out the cohort in LOCKSTEP
  **batched across personas** (one adapter handles all tags in a single `generate`/step). Returns
  the population ActivityLogs.
- Validate the WEIGHTED aggregate vs `funnel.py` anchors: overall conversion ≈ 0.083, S4/S6
  conditional churn within tol, ε ≤ 0.12 — AND that it tracks the GIVEN mix (90% franz → conv up;
  100% peter → down).
- **Success criterion (the hope):** each persona's per-step policy being faithful ⇒ mixing at the
  specified proportions reproduces the real funnel population statistically. A knob: set the mix
  → get a realistic, anchor-matching synthetic funnel. This is what the coach loop + demo run on.
- **DoD:** `generate_population(30/50/20, N=300)` → weighted ε ≤ 0.12 + conversion within tol;
  off-mix populations show the expected directional shift; fast (one batched model).

## Task 8 — Wire into the sim loop
- Swap chosen `LocalTeacher` into the sim/autoresearch loop (Mode A) and the coach loop
  (Mode B, `coach_widget` injection + `intervention_assessment`).
- **DoD:** Mode-A uplift run + Mode-B `coach_log` assessments produced locally, fast, offline.

---

## Execution order & checkpoints
`T1 → T2 → T3a (PROBE GATE) → T3 → T4 → T5 → T6 (GATE: ε≤0.12) → T7 → T8`

Two hard gates: **T3a** (state sampler actually yields balanced honest leaves before spending on
the full teacher run) and **T6** (ε passes before wiring into the loop).

## Risks
- Off-manifold sampled states → junk labels → constrain ranges / seed from teacher marginal (T2).
- Teacher cost/volume → bound M·K; probe first (T3a).
- Marginal still off after good per-step policy → that's expected; T6 calibration closes it.
- Prefill-bound batched eval → the smaller per-step prompt (T1) should also speed generation.
