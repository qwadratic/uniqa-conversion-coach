# UNIQA Conversion Coach — Build Journey

A first-person account of how the system was built, in the order it happened, with what
worked, what didn't, and why. The whole running system lives in **`sim_loop/`**.

---

## 1. First: the persona models

We started by modelling the **customer**, not the product. Three UNIQA segments —
**Judith** (Rising Hybrid), **Franz** (Online Affine), **Peter** (Service Affine) — became
**LLM personas**. Each persona's system prompt is `segment markdown + behavioural dials +
today's session instance` (`sim_loop/persona_prompt.py` over `prompts/personas/*`). The
persona carries a **mental state** (attention · satisfaction · effort_left · grasp ·
effort_vs_reward) that is threaded turn-to-turn, and at each funnel step it makes a *felt*
continue/leave decision — it leaves when a state variable crosses its tolerance. Grounded in
real UNIQA segmentation (n=4004) and the funnel anchors (S4 ≈66.7%, S5 ≈24%, S6+ ≈78% drop,
~5.6% conversion). Code: `sim_loop/persona.py`.

## 2. Second: the widget model

Then the **product**: a deterministic funnel state machine + screen renderer — the immutable
"app" the persona perceives. Eight steps S1→S8 (coverage → insured → personal info →
tariff/price → add-ons → personal data → health → review/purchase). It renders the per-step
screen, injects any coach widget, and advances; it is **not** an LLM. Code:
`sim_loop/widget.py` + `sim_loop/step_templates.json`.

## 3. Then: a good abstraction around the coach

The pivotal design move was to **abstract the coach away from the concrete funnel**. The coach
sees only the **observable event log** (no thoughts, no mental state, no persona label, no
health data) and emits **one typed effector / widget** (a JSON-render spec) under an annoyance
budget. Because it reasons over a *generic event trace* + a *typed effector library* (32
interventions × 7 categories × 10 frontend patterns × surfaces `on_page / email / whatsapp`),
the coach is **not tied to UNIQA's funnel** — the same detection→decision layer can sit on top
of **any** marketing service, funnel, or event stream. Conversion itself is **persona-dependent**
(the reward ρ): Judith = online *or* advisor handoff, Franz = online only, Peter = a qualified
service contact. Code: `sim_loop/coach.py`.

## 4. The baseline flow

Wiring persona ↔ widget ↔ coach into a turn loop (`sim_loop/run.py`) gave the baseline:
**anticipatory** coaching (the coach watches the trajectory and pre-places help on the wall the
persona is about to hit), an honest persona reaction (a *well-matched* widget lifts the relevant
state and can flip a would-be leave into a continue; a mismatched one annoys and is dismissed),
and a **persona-dependent success metric**. A paired A/B (coach off vs on) measures uplift, and a
JSON-rendered **replay demo** (`sim_loop/replay`, deployed live) shows the coach overlay floating
over the funnel, turn by turn.

## 5. Going local — the distillation attempt

With a baseline flow working, we tried to make a **local persona model** — replacing the prompted
LLMs with small local models (cheap, fast, private; the persona only runs at sim time, the coach
is the shipped product). We fine-tuned small LMs to compare two bases: **≈1B (MiniCPM5-1B)** and
**≈1.5B (Qwen2.5-1.5B)**, LoRA on the CINECA Leonardo A100 cluster.

**First round: training succeeded, but the student's *distribution* was lost.** We did hard-label
SFT on whole sessions. The student over-trained on the atomic steps where the persona *advances*
and decides **not** to abandon — and since each session has many positive "continue" steps but at
most **one** terminal event, most data points are positive. So the student learned to **always
convert** (judith conv = 1.00, churn = 0). It was unsuccessful.

**The known issue, and the fix.** This is a documented failure mode (class imbalance + exposure
bias + hard labels discarding the teacher's soft distribution — see
`docs/REPORT_distillation_collapse.md`). The solution is a big **synthetic dataset of atomic
points, not full sessions**: each data point is *(previous session events → the next-step
decision)*, sampled to **cover the state space** (including the leave-prone region) with **K
samples per context** so the empirical leave-rate *is* the soft target. With this per-step dataset
we restarted fine-tuning successfully — the **Judith + Franz Qwen2.5-1.5B adapters trained
cleanly** from `sim_loop`-derived data. **Then our compute quota / the reservation window closed**,
so the full two-base comparison is pending more allocation.

**The payoff insight (designed, not yet run).** If you **drop the full persona prompt from the SFT
input and replace it with a one-token persona TAG**, the student internalizes the static scaffold
and a **single tagged adapter** learns the behavioural patterns of *all three* personas while
still distinguishing them — one model, cross-persona batching. We couldn't experiment on this
before compute ran out.

**Conclusion:** theoretically **every LLM agent here — coach and personas — can be replaced with a
local model.** The path is proven end-to-end; only compute remains.

## 6. The self-improving coach (autoresearch) — the result

The coach's prompt is **not** hand-frozen. An autoresearch loop (`sim_loop/autoresearch.py`) edits
the `COACH_SYSTEM` prompt automatically against the persona simulator: it reads each round's
traces (which effector fired on which step → which outcome + the persona's reaction), an LLM
proposer writes **one targeted, auditable constraint**, a paired A/B against a shared coach-off
control measures **persona-success uplift**, and a gate keeps only real improvements
(`Δ̂ > incumbent + τ`, annoyance ≤ ceiling). **The current coach is the product of this loop, not
hand-tuning.** When a scale A/B exposed the coach mis-firing `form_simplify` on the price screen
and exhausting its budget before the price walls, the loop reads exactly those traces and proposes
the fix (`form_simplify` only on form steps; reserve budget for the final price step), keeping it
only if success actually rises.

---

## Status — proven vs pending

| | |
|---|---|
| **Proven** | persona↔widget↔coach turn loop · funnel-agnostic coach + 32-effector json-render demo (live) · persona-dependent conversion (ρ) · the per-step K-sampled distillation fix · Judith + Franz LoRA adapters trained from sim_loop's own data · the self-improving autoresearch loop |
| **Pending (compute only)** | the two-base (≈1B vs ≈1.5B) comparison · the single-tagged-model · scaled autoresearch to ship a winning coach prompt |

*Built at Zero One Hack, Vienna — synthetic-data-only, no live customer experimentation.*
