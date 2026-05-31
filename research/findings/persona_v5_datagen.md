# persona_v5 Dataset Generation — Findings

**Generated:** 2026-05-31  
**Method:** `sim_loop/run.py` (same code as main demo)  
**Arms:** coach-OFF (static widget) | coach-ON (32-effector json-render coach, budget=3)  
**Model:** `openai/gpt-4o-mini` (via OpenRouter, default)  
**Config:** `--sessions 120 --arms off,on --proportions real --coach-budget 3 --concurrency 16`  
**Output:** `datasets/persona_v5/`

---

## Run Logistics

| | |
|---|---|
| Smoke test | `--sessions 4`, 4-worker, confirmed both arms run cleanly |
| Call estimate (pre-launch) | OFF ≈ 600, ON ≈ 960 calls at concurrency 16 → ~4 min total |
| Actual wall time | OFF arm ~2 min, ON arm ~4 min; both well under 40-min hard limit |
| nohup launch | PID 51058; log `/tmp/persona_v5_run.log`; data written at END |
| Files written | `sessions_coach_off.jsonl` (876 KB), `sessions_coach_on.jsonl` (1.1 MB), `summary.json` |

---

## Session Counts & Persona Mix

Real-segment proportions applied; both arms share **identical persona assignment** (same pre-sampled sequence, seeded run).

| Persona | Segment share | Sessions (120 total) |
|---------|--------------|----------------------|
| judith  | 20.6%        | 15                   |
| franz   | 40.6%        | 50                   |
| peter   | 38.8%        | 55                   |

---

## Per-Variant Outcomes

### ARM OFF — Static Widget (coach disabled)

| Metric | Value |
|--------|-------|
| Sessions | 120 |
| Online convert | 18 (15.0%) |
| Advisor handoff | 18 (15.0%) |
| Abandon | 84 (70.0%) |
| **Persona-success rate** | **32 / 120 = 26.7%** |
| Coach interventions | 0 |

**Per-persona (OFF arm):**

| Persona | n | convert | advisor_handoff | abandon | Success\* | Success% |
|---------|---|---------|-----------------|---------|-----------|----------|
| judith  | 15 | 0       | 7               | 8       | 7         | 46.7%    |
| franz   | 50 | 14      | 4               | 32      | 14        | 28.0%    |
| peter   | 55 | 4       | 7               | 44      | 11        | 20.0%    |

\* Persona-success = outcome ∈ SUCCESS[persona]: Judith={convert,advisor\_handoff}, Franz={convert}, Peter={advisor\_handoff,convert}

### ARM ON — Coached Widget (coach active, budget=3)

| Metric | Value |
|--------|-------|
| Sessions | 120 |
| Online convert | 13 (10.8%) |
| Advisor handoff | 19 (15.8%) |
| Abandon | 88 (73.3%) |
| **Persona-success rate** | **29 / 120 = 24.2%** |
| Coach interventions | 256 total |

**Per-persona (ON arm):**

| Persona | n | convert | advisor_handoff | abandon | Success | Success% |
|---------|---|---------|-----------------|---------|---------|----------|
| judith  | 15 | 0       | 6               | 9       | 6       | 40.0%    |
| franz   | 50 | 11      | 3               | 36      | 11      | 22.0%    |
| peter   | 55 | 2       | 10              | 43      | 12      | 21.8%    |

---

## Coach Uplift (ON − OFF)

| Metric | OFF | ON | Δ |
|--------|-----|----|---|
| Online convert rate | 15.0% | 10.8% | **−4.2 pp** |
| Persona-success rate | 26.7% | 24.2% | **−2.5 pp** |
| Advisor handoff rate | 15.0% | 15.8% | +0.8 pp |
| Abandon rate | 70.0% | 73.3% | +3.3 pp |

**The coach arm produced negative uplift in this run.** The simulation's persona-rated interventions were mostly "helpful" at the micro level (see below), but the macro conversion metrics declined. Likely drivers:

1. **Budget exhaustion pattern**: 58/120 sessions hit budget=3 early (median exhaustion at S4–S5); many interventions fired on S3–S4 before the critical price-wall at S7–S8, leaving no budget for the sessions that actually needed it.
2. **form_simplify over-deployment on S4**: The coach fired `form_simplify` 53 times on `S4_TARIFF_SELECT` (a tariff-comparison screen, not a long form). This mismatch likely signalled irrelevance to Franz-type users and slightly increased friction.
3. **Franz users hurt most**: Franz success dropped 28% → 22% (−6 pp). Franz is online-affine and dislikes interruptions; `jump_to_pricing` and `form_simplify` at S3 may have added confusion rather than speed.
4. **Sample size caveat**: n=120 per arm; Δ of 2–4 pp is within plausible noise for this population size. Not statistically conclusive.

---

## Coached Arm — Intervention Assessment Distribution

Assessed reactions (from persona's `intervention_assessment.reaction` field):

| Reaction | Count | % |
|----------|-------|---|
| helpful | 170 | 67.2% |
| neutral | 78 | 30.8% |
| distracting | 3 | 1.2% |
| irrelevant | 2 | 0.8% |
| **Total assessed** | **253** | |

Engagement (persona emitted `widget_cta` = engaged vs `widget_dismiss`):

| | Count | % |
|---|-------|---|
| Engaged (True) | 157 | 62.1% |
| Not engaged (False) | 96 | 37.9% |

**Note**: High "helpful" ratings but negative macro uplift suggests the intervention\_assessment is soft self-report; it does not measure causal impact on the decision path. A persona can rate a widget "helpful" and still leave.

### Top Effectors Used

| Effector | Count | Engagement rate | Primary step(s) |
|----------|-------|-----------------|-----------------|
| form_simplify | 95 | 51.6% | S3, S4, S6 |
| jump_to_pricing | 58 | 63.2% | S3 |
| addon_skip_ok | 55 | 87.3% | S5 |
| quick_quiz | 13 | 30.8% | S4, S6 |
| preselect_optimal | 11 | 63.6% | S4 |
| price_preview | 7 | 71.4% | S4, S5, S6 |
| value_justification | 5 | 100.0% | S5 |
| callback_offer | 4 | 25.0% | S3, S6 |
| pricing_explain | 3 | 33.3% | S5, S6 |
| price_reframe | 2 | 50.0% | S5, S6 |

**Highest engagement**: `addon_skip_ok` at S5 (87.3%) and `value_justification` at S5 (100%). Lowest: `callback_offer` (25%) and `quick_quiz` (31%).

### Budget Usage Distribution

| Interventions used | Sessions |
|--------------------|----------|
| 0 | 4 |
| 1 | 34 |
| 2 | 24 |
| 3 (exhausted) | **58** (48%) |

Nearly half of coached sessions exhausted the full budget=3, most firing on S3–S5 before the critical price/health steps.

---

## SFT Pair Counts

Generated by `sim_loop/to_sft.py`:

| File | Sessions | SFT pairs | Pairs/session |
|------|----------|-----------|---------------|
| `datasets/persona_v5/static_sft.jsonl` | 120 | **697** | 5.8 avg |
| `datasets/persona_v5/coached_sft.jsonl` | 120 | **659** | 5.5 avg |

Each pair = one funnel step:
```json
{
  "messages": [
    {"role": "system",    "content": "<build_system_prompt(seg, session_instance)>"},
    {"role": "user",      "content": "<json.dumps(widget.render(...))>"},
    {"role": "assistant", "content": "<json.dumps(persona_output)>"}
  ]
}
```

The system prompt is rebuilt from `persona_prompt.build_system_prompt(seg, session_instance)` (identical to the one used at simulation time). The screen is reconstructed by replaying `widget.render` with the same state, history, and coach_injection from the recorded session.

---

## Class Imbalance — Leave/Continue Balance Note

### Overall decision distribution

| File | continue | leave | continue% | leave% |
|------|----------|-------|-----------|--------|
| static_sft.jsonl | 595 | 102 | 85.4% | 14.6% |
| coached_sft.jsonl | 552 | 107 | 83.8% | 16.2% |

**⚠️ Heavily imbalanced (~85% continue / ~15% leave).** This is expected for a real funnel trace (most steps see continue because sessions that leave early contribute fewer steps), but it is a known training pitfall.

### Per-step leave rates (static arm)

| Step | n | leave% |
|------|---|--------|
| S1_COVERAGE_TYPE | 120 | 0.0% |
| S2_INSURED_PERSONS | 120 | 0.0% |
| S3_PERSONAL_INFO | 120 | 8.3% |
| S4_TARIFF_SELECT | 110 | **34.5%** |
| S5_ADDON_SELECT | 72 | 4.2% |
| S6_PERSONAL_DATA | 69 | **33.3%** |
| S7_HEALTH_QUESTIONS | 46 | 13.0% |
| S8_REVIEW_PURCHASE | 40 | **55.0%** |

The funnel has three main churn points: **S4** (tariff price wall), **S6** (personal data form), and **S8** (final commitment). S1–S2 always produce `continue` (no churn in the persona model at those steps).

### Contrast with K-sampled approach (datagen_v2.py)

`research/datagen_v2.py` generates K independent turns per step with varied session parameters — each turn is independently sampled, so the leave/continue ratio is controlled by prompt design (typically ~30–40% leave). The persona_v5 sessions-as-trajectories approach produces **naturally lower leave rates** because:

1. Funnel trajectories: early steps gate later ones; sessions that would leave at S1 never appear at S8 → S8 "leave" rate is artificially low-absolute (only 40 sessions reach S8).
2. No K-oversampling per step: each session contributes exactly one SFT pair per step it traversed.
3. **Implication**: a model fine-tuned on persona_v5 SFT data in naïve cross-entropy will learn to continue too often. Recommended mitigations: (a) upsample leave-decision pairs (steps S4, S6, S8), (b) add synthetic "leave" turns from the K-sampled v2/v3/v4 datasets, or (c) use a weighted loss that boosts leave tokens.

---

## Files Produced

```
datasets/persona_v5/
├── sessions_coach_off.jsonl    876 KB  — 120 static sessions (raw sim output)
├── sessions_coach_on.jsonl    1.1 MB  — 120 coached sessions (raw sim output)
├── summary.json                        — run metadata + arm aggregate stats
├── static_sft.jsonl                   — 697 SFT pairs from static arm
└── coached_sft.jsonl                  — 659 SFT pairs from coached arm

sim_loop/to_sft.py                     — new converter (added; no other files changed)
```

---

## Recommendations

1. **Coach tuning needed**: `form_simplify` is dominating (37% of all acts); constrain the coach to use it only on S3/S6 (actual form steps). Add a step-appropriateness guard.
2. **Budget allocation**: budget=3 exhausted in 48% of sessions. Consider budget=2 with a "hold 1 for S7/S8" heuristic, or step-gated budgets.
3. **SFT class imbalance**: before fine-tuning on persona_v5 SFT, oversample leave-decision pairs from S4+S6+S8 or combine with the K-sampled v2–v4 datasets.
4. **Macro vs micro uplift gap**: the coach's micro-level "helpful" ratings do not translate to macro uplift in this run — the simulation faithfully models decision-level effects, but the coach's intervention selection and timing need improvement before online A/B testing.
5. **Next experiment**: re-run with `--coach-budget 2` + coach system-prompt patch limiting `form_simplify` to S3/S6 and `jump_to_pricing` to S3 only, to test if targeted restraint recovers the uplift.
