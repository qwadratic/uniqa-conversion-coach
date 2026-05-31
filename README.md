SEE DEMO LIVE: https://qwadratic.github.io/uniqa-conversion-coach/

# UNIQA Conversion Coach

> A detection + decision layer that sits on top of UNIQA's existing health-insurance
> calculator, watches the customer, and intervenes **only when it helps** — proven in
> simulation before it ever touches a real user.

---

## TL;DR (the casual version)

UNIQA's online calculator loses ~94% of people before they buy — mostly at two price
walls. We don't rebuild the calculator. We add a **Coach** that reads the user's
behaviour (hesitation, price-shock, tab-away to compare…), figures out **which of three
personas** they are, and shows the *one* right nudge — or stays quiet.

We can't A/B test on real customers mid-purchase, so we built a **simulator**: LLM-driven
**personas** click through a faithful **twin of the funnel**, and the **coach** reacts each
turn. The coach **improves itself** against this sim (propose a tweak → simulate → keep it
only if conversion goes up). The whole thing is honest *if and only if* the simulator
matches reality — and that match is **one measured number, ε**. Keep ε small and the
self-improvement is provably safe.

Three personas, three different "wins": **Judith** (online *or* a clean advisor handoff),
**Franz** (online only — pushing him to an advisor is a *failure*), **Peter** (a booked
callback *is* the conversion).

---

## The system at a glance

```
                 ┌───────────────────────────── one JSON contract ─────────────────────────────┐
                 │  events · effector commands · CoachObservation · CoachDecision · render spec  │
                 └──────────────────────────────────────────────────────────────────────────────┘
                          ▲                         ▲                          ▲
          ┌───────────────┴───────┐   ┌─────────────┴────────────┐   ┌─────────┴───────────────┐
          │   PERSONA  π_P         │   │  WIDGET / funnel  W       │   │   COACH  π_C            │
          │  (LLM, 3 segments)     │──▶│  (immutable app twin;     │──▶│  empty by default;      │
          │  acts on the screen    │   │   reacts: advance / no-op)│   │  else 1 effector/widget │
          │  reacts to effectors   │◀──│   coach lives INSIDE it   │◀──│  detects persona, picks │
          └────────────────────────┘   └───────────────────────────┘   │  the one right nudge    │
                                                                        └─────────────────────────┘
   turn loop:  persona acts → widget reacts → coach (after every turn) → persona reacts → repeat
               (persona may take several turns until something actually changes)

   ┌──────────── LOOP A · fast, synthetic, free ────────────┐   ┌──────── LOOP B · slow, real ────────┐
   │  π_P → W → π_C(θ) → Δ̂uplift → gate(Δ̂>τ) → θ'           │   │ prod logs → refit π_P (shrink ε)     │
   │        improve the COACH against the persona sim        │◀──│ off-policy eval π_C → re-run Loop A  │
   └────────────────────────────▲───────────────────────────┘   └─────────────────────────────────────┘
                                └─ valid ONLY while  ε ≤ τ/2  (the master invariant)
```

---

# Formal model

The system is a **typed stochastic process**: one immutable environment (the funnel) and
**two learnable policies** (persona, coach), coupled by a single scalar
**ε = distance(sim, reality)**. The coach self-improves *fast* against the persona sim
(Loop A); the persona is kept *faithful* against production (Loop B); **ε is the gate that
makes Loop A's wins real.** The whole system is self-contained in **`sim_loop/`**:
`sim_loop/run.py` (the turn loop + session JSON), `sim_loop/widget.py` (the funnel env + action
space), `sim_loop/persona.py` + `sim_loop/persona_prompt.py` (the persona), `sim_loop/coach.py`
(the coach + the 32-effector json-render library), and `sim_loop/autoresearch.py` (Loop A).

## 1. One formal object: `⟨ S, E, A, W, π_P, π_C, ρ ⟩`

| Symbol | Meaning | Code |
|---|---|---|
| **S** | funnel steps `S1…S7` + terminals `{convert, advisor_route, abandon}` | `sim_loop/widget.py` |
| **E** | event vocabulary (`EventType`) | `sim_loop/run.py` |
| **A(s)** | closed per-step action space `legal_events(s) × targets(s)` | `sim_loop/widget.py` |
| **W** | **environment** (immutable): `render(s)→JSON`, `apply(effector)→events`, `outcome()` | `sim_loop/widget.py` |
| **π_P** | **persona policy** `(s, history, coach_effector?) → Δevents + {continue\|leave}` | `sim_loop/persona.py` · `persona_prompt.py` |
| **π_C** | **coach policy** `obs → decision` (empty by default; 32-effector json-render) | `sim_loop/coach.py` |
| **ρ** | **success map** `(outcome, persona) → reward` — persona-dependent | `sim_loop/run.py` |

A **session** = a rollout alternating `π_P.step` and `π_C.decide` over `W` until a terminal
(= `sim.simulate(...)`). The three `@runtime_checkable Protocol`s — `PersonaModel`,
`WidgetModel`, `CoachModel` — are the only coupling; everything is swappable behind them.

**The turn model (exact).**
```
loop until terminal:
    e_P     = π_P.step(s, history)        # persona acts on the current screen
    s'      = W.transition(s, e_P)        # widget reacts: advance / change inputs / no-op
    obs     = W.observe(s', history, budget)
    d       = π_C.decide(obs)             # coach: NO_ACTION by default, else an effector/widget
    if d.acts: history += W.apply(d)      # effector mutates the widget AND enters π_P's next prompt
    # if the persona turn caused NO widget change AND the coach emitted NO effector,
    # the persona simply takes another turn (multi-turn advance) — that's how it progresses.
```
The coach is conceptually **inside** the widget (its optional reactive layer); the persona
only ever talks to the surface. The coach runs **after every persona turn** — empty
(`NO_ACTION`) by default, or it emits one effector/widget. An emitted effector appears in
the widget response the persona sees next turn and **perturbs the persona's running state**.

**Reward is persona-dependent:**
```
ρ(o, Judith) = 1  if o ∈ {convert_online, advisor_booked}
ρ(o, Franz)  = 1  if o = convert_online              (advisor = FAILURE)
ρ(o, Peter)  = 1  if o ∈ {service_contact, callback_booked}
```

## 2. The widget / intervention space — a typed grammar, never free text

```
CoachDecision = Surface × Intent × EffectorCommand × Render × Reasoning × Hypotheses
Surface  ∈ {on_page, email, whatsapp, landing, survey, advisor_booking}
Intent   ∈ CATALOG (sim_loop/coach.py)   Effector ∈ {NO_ACTION, SHOW_WIDGET, …}
```
Each **Intent is a typed record** `{id, addresses_pain, serves:{persona→target}, valid_steps,
valid_surfaces, fe_pattern, intrusiveness, copy_template, render_schema, guardrails}`, so
`decide()` is a **constrained argmax** over the catalog subject to: valid step/surface,
`EffectorCommand.validate()` (guardrails the **env** enforces regardless of policy — no
autofill on identity/health/consent; health/S6 never crosses a surface), intrusiveness ≤
temperature, and one shared annoyance budget. The persona emits `intervention_assessment`,
so every `(obs, decision, assessment, outcome)` is a labeled example for both policies.

## 3. The persona model — policy + ε + continuous eval

**Persona = a per-step Markov policy** `π_P(events, leave | step, running_state, disposition,
history, coach_effector?)`, parameterized by `θ_P = (segment tag, dials)`. The one number
that says "is the persona good":
```
ε(π_P) = TV( φ(D_sim), φ(D_real) )
φ = ( per-(persona,step) conditional churn, overall conversion, bounce-reason mix )
anchors (offline truth): S4 66.7% · S5 24% · S6+S7 ≈ 78% · conv 5.6% · mix 30/50/20
```
**Continuous persona evaluation** = a standing job emitting `eval_report.json{φ, ε, per-persona
conv, separability}` per dataset/model version, with a **promotion gate** (`ε ≤ ε_max`). Two
evolution mechanisms, kept separate: **make it cheap** (distill teacher LLM → one tagged
student LoRA) and **make it faithful** (re-fit dials/priors from prod logs, Loop B). Invariant:
**no anchor numbers in the prompt** — conformance must *emerge* from the state sampler +
calibration, or ε is circular. Why **K-sampling** (query the teacher K× per context, leave-rate
= soft target): the API has no logits, so K samples recover `P(leave|context)`; this distils the
teacher's *distribution* and breaks the 5:1 continue/leave imbalance that collapsed v1 to
"always convert". (See `docs/REPORT_distillation_collapse.md`.)

## 4. The coach model — autoresearch operator + gate

```
U(θ_C) = E[ ρ(o,persona) − λ·annoyance − μ·cost ] ;   Δuplift = U(θ_C) − U(∅)
PROPOSE θ' = mutate(θ_C)  →  SIMULATE N paired sessions/persona  →  Δ̂ = Û_sim(θ') − Û_sim(θ_C)
GATE    accept ⇔ Δ̂ > τ ∧ annoyance ≤ ceiling
```
The gate is where ε re-enters. **Empirical now**: accept iff `Δ̂ > τ`. **Formal (Z3,
deferred)**: if `ε` small ⇒ `b = L·ε`, and `τ ≥ 2b`, then
`U_real(θ') − U_real(θ_C) ≥ Δ̂ − 2b > 0` — every accept is a real improvement (T1), monotone
(T3), convergent (T4). The 1B coach distillation (MiniCPM5-1B LoRA on Leonardo) is a separate
gated step: ship only if `U_sim(FT) ≥ U_sim(prompt)`.

## 5. The coupling — two loops, one invariant

> **Master invariant:** *never ship a coach whose synthetic gain Δ̂ isn't backed by a freshly
> measured `ε ≤ τ/2`. Loop B keeps ε small; Loop A maximizes Δ̂; the gate is the only place
> they meet.* Everything else (multi-surface, `p_identify`, feature importance) hangs off this.

## 6. What makes it executable (gaps)

1. A typed **`Ledger`** — append-only JSON both loops write per gate decision
   `{θ_C, θ_P, ε, Δ̂_sim, Δ_real(IPS), τ, accepted?}`.
2. **ε as a first-class artifact** — `sim_loop/run.py` computing `φ`, `ε`, auto-`τ`.
3. **Typed catalog** — `sim_loop/coach.py` as the `Intervention` record (§2).
4. **`SurfaceModel` Protocol** + `CoachDecision.surface` + shared cross-surface budget.
5. **IPS off-policy estimator** in `sim_loop/` — the honest bridge Δ̂_sim → Δ_real.

---

# Autoresearch — the self-improving coach (Loop A)

The coach is **LLM-driven**: its whole policy is the `COACH_SYSTEM` prompt in `sim_loop/coach.py`.
Autoresearch (`sim_loop/autoresearch.py`) improves **that prompt** automatically, against the
persona simulator — the current coach is the product of this loop, not hand-tuning.

### How it works
```
PROPOSE  an LLM proposer reads the last round's traces (which effector fired on which step →
         which outcome + the persona's intervention_assessment) and writes ONE targeted,
         auditable constraint into the COACH_SYSTEM prompt (e.g. “never use form_simplify
         outside S3/S6”, “hold ≥1 budget for the final price step”). Curated fallbacks encode
         known fixes if the LLM call fails.
SIMULATE N paired sessions through the persona sim, against a coach-OFF control run ONCE
         (shared baseline → every variant is directly comparable, no re-paying for control).
EVALUATE Δ̂ = persona-SUCCESS(variant) − success(control)   [ρ is persona-dependent, see §1]
GATE     accept iff  Δ̂ > incumbent_uplift + τ  AND  annoyance ≤ ceiling
LEDGER   every round appended to ledger.jsonl; the winner saved to best_prompt.txt
```
The empirical gate (`Δ̂ > τ`) is the shippable stand-in for the deferred Z3 certificate; the
incumbent is monotone (only ever replaced by a strictly-better prompt).

### Run it
```bash
python sim_loop/autoresearch.py --rounds 6 --sessions 30 --tau 0.025 \
       --proposer llm --concurrency 8 --out sim_loop/out/autoresearch_v1
# artifacts: ledger.jsonl · best_prompt.txt (paste into sim_loop/coach.py) · report.md
```
It prints a call estimate first and is multithreaded; size `--sessions` for the signal you
need (n=30, τ≈0.025 is the sweet spot) and never a job over ~15 min.

### Example — latest improvement
A scale A/B (persona_v5, n=120 real mix) showed the coach had gone **negative** (−4.2pp online,
−2.5pp persona-success): it over-fired `form_simplify` on **S4** (a price screen, 37% of all
acts) and **exhausted its budget** before the real price walls (S7/S8), hurting Franz most.
Autoresearch reads exactly those traces and proposes the fix — e.g. *“NEVER use form_simplify
outside S3/S6”* and *“reserve ≥1 intervention for the final price step”* — and the gate keeps it
only if persona-success actually rises. That trace-→constraint loop is how the coach prompt
earns each line it carries.

---

# Distillation — local persona & coach models

Every agent here (persona **and** coach) is a prompted LLM over OpenRouter. Distillation replaces
them with **small local models** (cheap, low-latency, private, no API) — the persona only needs
to run at sim time, the coach is the shipped product.

### Data — from the SAME code as the demo
`sim_loop/to_sft.py` builds SFT pairs by **replaying sim_loop's own builders** over recorded
sessions: `system = persona_prompt.build_system_prompt(seg, session_instance)`, `user =
widget.render(...)`, `assistant = the recorded persona_output`. So the training data is generated
by the identical process that runs the live demo — two variants: **static** (coach-off arm) and
**coached** (coach-on arm).
```bash
python sim_loop/run.py --sessions 120 --arms off,on --concurrency 16 --out datasets/persona_v5
python sim_loop/to_sft.py        # → {static_sft.jsonl, coached_sft.jsonl}  ({system,user,assistant})
```

### ⚠️ The collapse, and how to avoid it (the issue we hit)
Our first attempt — **hard-label SFT on whole sessions** — collapsed: the student learned to
**always convert** (judith conv=1.00, S4/S6 churn=0, ε≈0.48 vs the teacher's ≈0.10). Three
well-known modes compounded (see `docs/REPORT_distillation_collapse.md`):
1. **Class imbalance** — a session has ~4–5 `continue` steps and **at most one** terminal `leave`
   (~85/15 continue:leave), so the model modes to “continue → convert”.
2. **Exposure bias / state drift** — SFT only sees teacher-visited states (mostly “continue”); at
   inference the student visits its own states and compounds the bias (DAgger).
3. **Hard labels discard the soft distribution** — one sample per step throws away `P(leave)`,
   which is the entire point of a persona (Hinton 2015).

**Fix — per-step, state-covering, K-sampled atomic data** (`research/datagen_v2.py`). Instead of
full trajectories, sample **(prior context → next-step decision)** atomic points that COVER the
state space (including the leave-prone region), and query the teacher **K× per context** so the
empirical leave-rate **is** the soft target. This rebalances the data and distils the teacher's
distribution. Targets (the 66/24/78 funnel anchors) stay **eval-only**, never in the prompt.

### Reproduce the fine-tuning (Leonardo HPC, `slurm/`)
The Leonardo connect skill is **vendored in this repo** at `skills/leonardo-connect/` (SSH via
`expect`, credentials from `.env`: `LEONARDO_USERNAME` / `LEONARDO_PASSWORD`).
```bash
# 0) DOWNLOAD the base model from HuggingFace — on a LOGIN node (compute nodes have NO internet):
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/qwen2.5-1.5b
#    (≈1B base alternative)  huggingface-cli download openbmb/MiniCPM3-1B --local-dir ~/models/minicpm-1b

# 1) GENERATE the fine-tuning dataset from the SAME sim_loop code, then build SFT pairs:
python sim_loop/run.py --sessions 200 --arms off,on --concurrency 16 --out sim_loop/out
python sim_loop/to_sft.py                         # → {system,user,assistant} jsonl in slurm/data_sim/

# 2) TRANSFER to Leonardo — login-node scp is BLOCKED; use the DATAMOVER with ABSOLUTE paths,
#    and stage big files under $SCRATCH (/leonardo_scratch/large/usertrain/<user>):
tar czf sft.tgz slurm/data_sim
scp sft.tgz <user>@dmover1.leonardo.cineca.it:/leonardo_scratch/large/usertrain/<user>/
#    then, on a login node:  tar xzf $SCRATCH/sft.tgz -C ~/zero-one/

# 3) CONNECT + SUBMIT the LoRA job (Qwen2.5-1.5B / MiniCPM base · r=16 · bf16 · FA2):
bash skills/leonardo-connect/scripts/leo.sh        # interactive login (creds from .env)
sbatch slurm/slurm_persona_sim.sh                  # partition boost_usr_prod, reservation s_tra_ncc
```
**Avoid the traps we hit:** compute nodes have **no internet** → pre-stage base weights on a
login node; **never `scancel` mid-train** (the adapter is saved only at the end — recover from
`checkpoint-N/`); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and **batch ≈48** (100 OOMs
on 64GB); use a **completion-only collator with the right response template** (`assistant_only_loss`
silently masked the *entire* sequence for one persona → loss=0, degenerate adapter); trust
`mean_token_accuracy`, not `eval_loss` (which goes `nan` under completion masking).

### One model for all personas (the tag trick)
If you **drop the full persona prompt from the SFT input and replace it with a one-token persona
TAG**, the student internalizes the static scaffold and a **single tagged adapter** learns all
three personas while still separating them (`research/student_view.py` measured 7.4–17.9× prompt
compression). This enables cross-persona batching and one-model serving. *Designed, not yet
validated* — compute ran out before the experiment.

### Status
The pipeline ran end-to-end on Leonardo from sim_loop-derived data: **Judith + Franz Qwen2.5-1.5B
LoRA adapters trained successfully** inside the reservation window; the per-step K-sampled fix is
in place. The compute quota then closed, so the full two-base (≈1B MiniCPM vs ≈1.5B Qwen) comparison
and the single-tagged-model run are pending more allocation — but the path is proven: **every
LLM agent here can be made local.**

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate          # Python 3.11+
echo 'OPENROUTER_API_KEY=sk-or-...' > .env                  # the ONLY requirement for the demo engine
# pip install -r requirements.txt                           # optional — only for LOCAL fine-tuning / HF download

# proven paired A/B (coach off vs on) — the main demo's own engine
python sim_loop/run.py --sessions 50 --arms off,on --proportions real --coach-budget 3 --concurrency 12 --out sim_loop/out

# coach-prompt autoresearch (Loop A): propose → simulate → gate → ledger
python sim_loop/autoresearch.py --rounds 8 --sessions 20 --tau 0.03 --concurrency 8

# build distillation SFT from the SAME sim_loop code, then LoRA on Leonardo (slurm/)
python sim_loop/to_sft.py

# interactive replay demo (Vite + React + TS)
cd sim_loop/replay && npm install && npm run dev
```
Needs `OPENROUTER_API_KEY` in `.env` (persona + coach are LLM-driven). Live demo:
**https://qwadratic.github.io/uniqa-conversion-coach/**

## Layout

```
sim_loop/                the whole system — self-contained engine + demo
  run.py                 turn loop: persona ↔ widget ↔ coach (the A/B engine, ρ, success metric)
  widget.py              immutable funnel state machine + screen renderer (8 steps)
  persona.py             LLM persona (state threaded turn→turn)
  persona_prompt.py      persona system prompt (segment md + behavioural dials + session instance)
  coach.py               coach policy + the 32-effector JSON-RENDER library (detection→decision prompt)
  autoresearch.py        Loop A: propose coach-prompt variant → simulate → gate → ledger
  to_sft.py              sim_loop sessions → SFT pairs (the distillation data, same code as the demo)
  step_templates.json    canonical per-step screens · session_pools.json  session-instance pools
  replay/                Vite + React + TS demo — the coach overlay JSON-rendered over the funnel
slurm/                   Leonardo HPC — LoRA fine-tune (Qwen2.5-1.5B) on sim_loop-derived SFT
prompts/personas/        persona system prompts (+ behavioural dials)
docs/                    see below
```
Generated data (`datasets/`, `models/`, `slurm/data*`, `slurm/out`, `sim_loop/out`) and the legacy
package (`calculator/ coach/ persona/ …`, superseded by `sim_loop/`) are gitignored — kept on disk.

## Docs

| Document | Contents |
|---|---|
| [`REPORT.md`](REPORT.md) | **The build journey** — personas → widget → coach abstraction → baseline → local-model distillation (the collapse + fix) → self-improving coach. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The `sim_loop/` architecture: the three roles, the turn loop, the two loops, the distillation bridge. |
| [`docs/COACH_MODEL.md`](docs/COACH_MODEL.md) | The Coach: detection, decision workflow, signals, hypotheses (H1…), per-screen interventions. |
| [`docs/FUNNEL_AUTOPSY.md`](docs/FUNNEL_AUTOPSY.md) | The real UNIQA funnel, screen by screen, and where users drop off (grounds `sim_loop/widget.py`). |
| [`docs/PERSONA_DISTILL_V2_PLAN.md`](docs/PERSONA_DISTILL_V2_PLAN.md) · [`docs/REPORT_distillation_collapse.md`](docs/REPORT_distillation_collapse.md) | The per-step K-sampled distillation plan + the v1-collapse → fix writeup. |

> Every doc describes the system as implemented in **`sim_loop/`**. Earlier docs/packages
> (the first-cut `calculator/`/`coach/`/`persona/` package, `PIPELINE_PLAN`, `ROADMAP_FINAL`,
> `WIDGET_TWIN_DESIGN`, …) are gitignored — kept on disk, out of the repo.

---

*Synthetic-data-only validation — no live customer experimentation. Conversion is defined
per the track's online-completion scope.*
