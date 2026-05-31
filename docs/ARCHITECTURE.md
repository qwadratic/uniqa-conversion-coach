# Architecture — App / Coach split, contracts, and the dual learning loop

How the pieces fit, what the Coach can actually do, and the two processes that
make it improve. Backed by code: `contracts.py`, `coach_io.py`, `psyche.py`,
`autoresearch.py`, `sim_loop/autoresearch.py`.

---

## 1. How persona models work against the journey *today*

Two engines exist; both treat the Coach as a fixed rule.

**Psyche engine** (the live one — drives demo + autoresearch):

```
init_mind(persona) ─▶ Mind{intent, 6 latent vars}
  for each step:
    step_dynamics(mind, step)        # latents evolve (price arrives, effort drains)
    signals = generate_signals(...)  # behaviour proxy emitted
    action  = decide_action(signals) # ← FIXED rule coach
    apply_coach_effect(mind, action) # coach mutates the MIND (↑price_readiness…)
    evaluate_bounce(mind, step)      # hazards combine → bounce reason or advance
```

The persona is a latent `Mind`; the Coach acts *on the mind*, then the outcome is
re-rolled. Honest, multi-causal. **Limitation:** the Coach is hand-written
`decide_action`, and it reads `signals` + `persona` directly. That's what we now
change.

---

## 2. The split: one immutable component, one mutable component

```
        ┌──────────────────────────────┐   activity log (events)   ┌────────────┐
        │  APP  — IMMUTABLE             │ ─────────────────────────▶│  COACH     │
        │  • 11-step form state machine │                           │  MUTABLE   │
        │  • renders screens (JSON)     │   effector cmd + reasoning │  (trained) │
        │  • emits Events               │ ◀─────────────────────────│            │
        │  • executes Effector commands │                           └────────────┘
        └──────────────────────────────┘
                    ▲
                    │ (in simulation) the PERSONA MODEL plays the user:
                    └── psyche Mind → behaviour → Events
```

- **APP (never trained).** Fixed surface: the in-scope Privatarzt funnel
  (`funnel.py` + `scope.py`), a fixed **effector API**, and a JSON renderer. The
  Streamlit demo and a future React app are two renderers of the same envelopes.
- **COACH (the only thing we train).** A policy `π(action | observation)`.
  Input = user **activity log**. Output = one **effector command** (or
  `NO_ACTION`) + **human-readable reasoning** + **testable hypotheses**.
- **PERSONA MODEL (the learned simulator).** In simulation it stands in for the
  real user: `psyche.Mind` → behaviour → Events. It is re-fit from real data in
  the online loop (§6).

Contracts live in `contracts.py`; the simulation round-trip is proven in
`coach_io.py` (+ `test_contracts.py`).

---

## 3. What the Coach can actually do (capability surface)

Two layers, cleanly separated so we train *intent* against a *fixed* mechanism.

**Effectors** — the APP's fixed mechanical capabilities (`contracts.Effector`):

| Effector | Does | Guardrail |
|---|---|---|
| `NO_ACTION` | stay silent | the most-used action; annoyance control |
| `SHOW_WIDGET` | overlay a coach card (payload = intent + copy) | counts against budget (max 3) |
| `FOCUS_FIELD` | move focus to a field | passive |
| `SCROLL_TO` | scroll viewport to an element | passive |
| `HIGHLIGHT` | emphasise an element | passive |
| `AUTOFILL` | fill a field with the user's known value | **forbidden** on identity/health/consent (`NEVER_AUTOFILL`) |
| `FILL_SAMPLE` | fill placeholder data so the user can click through and explore | only on `coverage/insured/tariff` (`SAMPLE_FILLABLE`) |
| `PRESELECT_TARIFF` | pre-select Start/Optimal (online-completable) | online tariffs only |
| `SAVE_PROGRESS` | capture email → resume-later / channel handoff | consent-gated |

**Intents** — the trainable strategy (`coach.CoachAction`: `price_reframe`,
`upgrade_explain`, `health_explain`, `form_helper`, `progress_saver`, …). An
intent is realised through an effector (mostly `SHOW_WIDGET`); `coach_io.INTENT_EFFECTOR`
maps the two. This is why the action space is both *expressive* and *safe*: the
app validates every effector command (`EffectorCommand.validate()`) no matter what
the policy proposes. Detection/“read signals from the user” is implicit: the Coach
*only* sees the activity log, so reading signals is its whole input.

---

## 4. The Coach I/O contract (what we train)

```python
observe: CoachObservation {           # contracts.CoachObservation
   session_id, step,
   activity:  [Event…],               # the user activity log window
   form_state: {field: filled/valid}, # NO values, NO latent persona/intent
   budget_remaining: int
}
        │  CoachModel.decide(obs)
        ▼
CoachDecision {                        # contracts.CoachDecision
   command:    EffectorCommand,        # the action token we optimise (or NO_ACTION)
   reasoning:  str,                    # MANDATORY, human-readable
   hypotheses: [Hypothesis…],          # falsifiable beliefs about the user
   confidence, value_estimate          # π's predicted P(convert | act)
}
```

Key properties:
- The observation **never** contains the persona label or latent mind — the Coach
  must *infer* segment/state from behaviour, like in production.
- `reasoning` is required (auditability + the design ask).
- `RuleCoachModel` is today's baseline; a learned policy swaps in behind the same
  `decide(obs) -> CoachDecision` signature with zero downstream change.

### Hypothesis validation (the bridge to learning)

A `Hypothesis` is a falsifiable belief: *claim* (“user is price-shocked”), the
*latent* it implies (`price_readiness`), a probability `p`, and a **prediction**
(“if true and we don’t act → ABANDON”). After the journey continues,
`score_hypotheses()` marks each confirmed/refuted. The hit-rate:
1. grades the Coach’s **model of the user**, and
2. is the supervised signal that re-fits the **persona model** in the online loop.

---

## 5. JSON-render everything (it’s an app, not a Streamlit toy)

Every object is JSON-serialisable (`test_contracts.py` round-trips them). The
frontend consumes one message type, `RenderEnvelope{kind, step, spec, hud?}`,
where `kind ∈ {step_screen, coach_widget, effector, outcome}`. Streamlit renders
it today; a React app renders the identical JSON tomorrow. The simulation emits
the same `ActivityLog` + `CoachDecision` JSON a real browser SDK would — so
“prove it in simulation” and “ship the app” use one contract.

---

## 6. The two learning processes

### Loop A — Synthetic autoresearch (runs forever)

```
PERSONA MODEL ─▶ APP ─▶ COACH(π) ─▶ outcome ─▶ gate(Δuplift > τ) ─▶ π'
   (learned simulator)                          (Z3-certified)
```

Model-based RL: the persona model is the learned environment; reward =
`conversion − annoyance_penalty − intervention_cost`. `autoresearch.py` already
implements the gated hill-climb; `sim_loop/autoresearch.py` certifies that
**if** the simulator is faithful (`|U_sim − U_real| ≤ b`, `τ ≥ 2b`) **then** every
accepted policy is a real improvement, monotone and convergent. This loop is cheap
and never needs real users — run it on CPU forever (Leonardo `L2`).

### Loop B — Online feedback (periodic, real data)

Every batch of real `(activity_log, coach_decision, outcome)` tuples:

```
real batch ─┬─▶ (1) RE-FIT persona model    : fit psyche latents/intent-mix so
            │        synthetic funnel stats match real → shrinks ε (assumption A1)
            ├─▶ (2) OFF-POLICY EVAL the coach: IPS/counterfactual estimate of the
            │        live policy on real logs → ground-truth vs synthetic estimate
            ├─▶ (3) hypothesis hit-rate      : grade + recalibrate the user model
            └─▶ (4) RE-RUN Loop A on the improved simulator → propose π'
                     ─▶ Z3 gate ─▶ shadow deploy ─▶ next batch
```

Loop B’s entire job is to **keep ε small** so Loop A’s proof stays valid. RL
framing: Loop A = unlimited synthetic policy improvement under a learned model;
Loop B = periodic model re-calibration + honest off-policy policy evaluation on
real traffic. The Z3 certificate is the safety rail between them — a synthetic
improvement only ships if the simulator it was found on is close enough to reality.

```
                 cheap, infinite                      expensive, periodic
        ┌───────────── Loop A ─────────────┐   ┌──────────── Loop B ────────────┐
        │ persona model → coach → gate → π'│ ◀─│ real data → re-fit persona,     │
        │ (Z3-certified, runs on CPU 24/7) │   │ off-policy eval, then re-run A  │
        └──────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 7. Status / what’s real now

- ✅ Contracts (`contracts.py`): events, effectors+guardrails, hypotheses, Coach I/O, render envelope — all JSON round-tripped.
- ✅ Adapter (`coach_io.py`): psyche signals → activity log → observation → `RuleCoachModel.decide` → decision; hypothesis scoring.
- ✅ Loop A: `autoresearch.py` (empirical gate `Δuplift > τ`). Formal Z3 certificate **deferred** (`sim_loop/autoresearch.py`).
- ▢ Loop B: interfaces designed here; `PersonaFit` (re-fit latents) + IPS off-policy eval are the next build. No real data yet → ε estimated from calibration anchors.
- ▢ Wire `RuleCoachModel` into `journey.run_journey` as the decision source (drop-in for `decide_action`) so the whole sim runs on the contract.
- ▢ Swap `RuleCoachModel` body for a learned policy (same signature).

93 tests passing.

---

## 8. UI design tokens (for the renderer)

Every coach widget is JSON (§5); the renderer styles it with UNIQA's visual
identity so interventions read as "from UNIQA". Tokens inferred from UNIQA's
public funnel + brand guidelines.

| Token | Value | Use |
|---|---|---|
| `color.primary` | `#0046A0` (UNIQA blue) | Headings, primary CTA, progress fill |
| `color.primary.dark` | `#002D6A` | Hover/active CTA |
| `color.accent` | `#E2001A` (UNIQA red) | Urgency, "save" badges, hesitation flag |
| `color.success` | `#1FA971` | Coverage included, confirmations |
| `color.warning` | `#F0A028` | "Advisory required" tags, soft alerts |
| `color.surface` / `.alt` | `#FFFFFF` / `#F4F6FA` | Card bg / page bg |
| `color.ink` / `.muted` | `#1A1F2C` / `#5C6479` | Body / captions |
| `color.border` | `#D6DBE5` | Card borders, dividers |
| `radius.card` / `.pill` | `12px` / `999px` | Cards / badges |
| `shadow.card` | `0 2px 8px rgba(0,0,0,.06)` | Resting cards |
| `shadow.elevated` | `0 8px 24px rgba(0,70,160,.12)` | Coach intervention widgets |
| `font.family` | `"UNIQA Sans", "Inter", system-ui` | Whole product |
| `font.numeric` | tabular-nums | All price displays |

**Tone of voice:** Sie-form (formal German), concrete numbers first /
reassurance second, no marketing superlatives (soft proof: "seit 1811",
"AAA-rated"), Austrian conservatism (disclaimers visible), no emojis, no
exclamation marks except rare urgency. Brand tone vector (maps to widget `tone`
props): trustworthy 0.95 · clear 0.90 · austrian 0.85 · digital_forward 0.80 ·
warm 0.55 · urgency_default 0.20.
