# Architecture — `sim_loop/`

The whole system is **self-contained in `sim_loop/`**: three roles (persona, widget, coach)
talking over one session-JSON contract, a turn loop that runs them, a self-improving
autoresearch loop on the coach, and a converter to distillation data. This doc maps the code;
the abstract formal model is in the top-level `README.md`, the build journey in `REPORT.md`.

```
                       ┌──────────────────────── one session-JSON contract ────────────────────────┐
                       │  per step: { persona_output, coach_decision, shown_coach }                 │
                       └───────────────────────────────────────────────────────────────────────────┘
   sim_loop/persona.py ─────────▶  sim_loop/widget.py  ─────────▶  sim_loop/coach.py
   LLM persona (state              immutable funnel state          coach policy (empty by default);
   threaded turn→turn)             machine + screen renderer        one json-render effector/widget
        ▲   reacts to the coach widget shown on the screen   │ effector pre-placed on the next screen
        └───────────────────────────────────────────────────┘
                            orchestrated by sim_loop/run.py (the turn loop)
```

## The three roles

| Role | File | What it is |
|---|---|---|
| **Persona** `π_P` | `persona.py` + `persona_prompt.py` | LLM. System prompt = `segment.md` + behavioural dials + session instance (set once). Mental state (attention/satisfaction/effort_left/grasp/effort_vs_reward) threaded turn→turn. Emits `events + decision(continue\|leave) + new state + feeling` each step. |
| **Widget** `W` | `widget.py` + `step_templates.json` | Deterministic funnel state machine + per-step screen renderer (S1→S8). **No LLM.** Injects any coach widget into the screen as `coach_intervention_shown`; advances/terminates. The immutable "app". |
| **Coach** `π_C` | `coach.py` | LLM policy. Sees only the **filtered** event log (no thoughts/state/persona/health). `EFFECTOR_LIBRARY` = 32 interventions (7 categories × 10 fe_patterns × surfaces). Emits one json-render `command` {effector, fe_pattern, surface, title, body, cta} or `NO_ACTION`. Mode `skip` = the control arm. Policy = the `COACH_SYSTEM` prompt (overridable via `system=`). |

## The turn loop (`run.py`)

```
for step in S1..S8:
    coach.decide(log_so_far, step)            # ANTICIPATORY: pre-place help on THIS screen
    screen = widget.render(step, state, history, session_instance, intent, coach_injection)
    persona.step(screen)                      # persona acts + reacts to the widget; may LEAVE
    classify_outcome(...)                      # convert | advisor_handoff | abandon
```
- **Anticipatory coaching:** the coach is consulted *before* the persona's decision, so a
  well-timed widget is present *when* the persona decides the step it's about to bounce on
  (not uselessly on the next screen).
- **Persona-dependent conversion** (`ρ`, the `SUCCESS` map): Judith = {convert, advisor_handoff},
  Franz = {convert}, Peter = {advisor_handoff, convert}. The arm summary reports both online
  `convert_rate` and persona `success_rate`, and the off↔on **uplift**.
- **Information isolation:** the coach never sees the persona's thoughts, state, feeling, label,
  or S6 health data — only the observable event log (`filtered_event`).

## The two loops

- **Loop A — autoresearch (`autoresearch.py`).** Improves the `COACH_SYSTEM` prompt against the
  sim: an LLM proposer reads round traces → one targeted constraint → paired A/B vs a shared
  coach-off control → gate on persona-success uplift (`Δ̂ > incumbent + τ`, annoyance ≤ ceiling)
  → `ledger.jsonl` + `best_prompt.txt`. (README → "Autoresearch".)
- **Loop B — re-grounding (design).** Periodically re-fit the persona priors and off-policy-eval
  the coach on real logs to keep the simulator faithful. The master invariant: *never ship a
  coach whose synthetic gain isn't backed by a faithful simulator.*

## Distillation bridge (`to_sft.py`)

`to_sft.py` turns recorded sessions into SFT pairs by **replaying the same builders the demo
uses** — `system = persona_prompt.build_system_prompt(...)`, `user = widget.render(...)`,
`assistant = the recorded persona_output`. Two variants: **static** (coach-off) and **coached**
(coach-on). Those pairs feed the LoRA training in `slurm/` (Leonardo). See README → "Distillation".

## The demo (`replay/`)

A Vite + React + TypeScript app that replays the A/B sessions: a faux funnel form with the coach
**json-rendered overlay** (a lightweight `fe_pattern → component` registry) floating on top —
deliberately a distinct visual layer from the funnel. Deployed to GitHub Pages.

## Files

```
sim_loop/
  run.py                turn loop · classify_outcome · SUCCESS (ρ) · paired A/B summary · CLI
  widget.py             funnel state machine + screen renderer + coach injection
  persona.py            LLM persona (state threaded turn→turn)
  persona_prompt.py     persona system prompt (segment md + dials + session instance)
  coach.py              coach policy + the 32-effector json-render EFFECTOR_LIBRARY + COACH_SYSTEM
  autoresearch.py       Loop A: propose coach-prompt edit → simulate → gate → ledger
  to_sft.py             sessions → SFT pairs (same builders as the demo)
  llm.py                OpenRouter client (stdlib urllib) + JSON extractor
  step_templates.json   canonical per-step screens   ·   session_pools.json  session-instance pools
  replay/               Vite + React + TS demo (coach overlay json-rendered over the funnel)
```
