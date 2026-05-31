# Persona Playbook — recording human sessions + emergent statistical targets

How to hand-record realistic human sessions for Judith / Franz / Peter (convert **and**
churn), so we have a **behavioral + timing ground truth** to calibrate the LLM persona
agent against — and later the fine-tuned local persona models.

> **Iron rule — targets are MEASURED, never SCRIPTED into any agent.**
> The conversion/bounce anchors (S4 ≈ 66 %, S6 ≈ 78 %, overall ≈ 5.6 %, mix 30/50/20)
> live **only** in `funnel.py` as the eval reference. They must **never** appear in a
> persona system prompt or a training example. We tune the *prompt* and the *training
> set* until those numbers **emerge** from behavior, then we score against the anchors.
> (See §5 — the persona `.md` briefings currently leak the percentages; they get
> sanitized before use.)

---

## 0. Where to record

Capture app: `sim_loop/replay/` (Vite + React). Two surfaces, same `Recorder` / same JSON schema
(`contracts.ActivityLog`, every event inside `widget.legal_events(step)`):

- **`?mode=twin`** — the 1-1 funnel twin (recommended; richest, parity-tested action space).
- default mode — the classic capture screen (simpler; fine for quick takes).

```bash
cd webapp && npm run dev          # open http://localhost:5173/?mode=twin
```

Per session: pick the persona in the sidebar, **behave as that persona** (do not perform a
number), then end by converting or leaving. Download the log JSON → save under
`_local/captures/<persona>/<convert|churn>_<n>.json`. Real mouse movement, dwell→hover,
tab-switching and idle gaps are captured automatically — **move the mouse like the persona
would**, don't just click.

The UI events you control (all already in the legal vocabulary):

| Step | What you can do | Emits |
|---|---|---|
| S1 coverage | click **Bei Arztbesuchen** (in-scope) or **Im Krankenhaus** (→ advisor exit) | `select` / `abandon advisor_route(hospital)` |
| S2 insured | **Ich selbst** (in-scope) or **Andere Personen** (→ advisor exit) | `select` / `abandon advisor_route(others)` |
| S3 personal | type DOB, open SV dropdown, filter, pick | `keystroke` `dropdown_open` `select` |
| S4 tariff | click **Start/Optimal** (online → reveals price, advances); click **Opt.Plus/Premium** (advisory-only, does **not** advance); hover a tariff (price_hover); hover (i) rows | `select`+`price_reveal` / `premium_click` / `price_hover` / `tooltip_open` |
| S6 personal+health | type fields, set health **no/yes** (yes → price jump), calc → final price, **Abschließen** | `keystroke` `price_reveal` `submit` `convert` |
| anywhere | leave via external link / close page; switch browser tab; sit idle | `abandon <reason>` / `tab_blur`+`tab_focus` / `idle` |

Online-completable path = **Bei Arztbesuchen → Ich selbst → Start or Optimal → finish**.
Everything else (hospital, other-persons, Opt.Plus/Premium) routes to an advisor = an
online abandon.

---

## 1. Judith — Rising Hybrid (30 %)

**Identity:** 43, Vienna, mid-management, busy, educated. Researches online, but wants a
trusted human at the moment of commitment. Disengages **silently** when frustrated; says
"I'll think about it" even when the real reason is specific.

**Arrival context (1st thought):** came on purpose — a trigger (recent hospital visit, a
child needs better cover, a colleague mentioned UNIQA). Googled "private Krankenversicherung
Wien", wants a price estimate **before** talking to anyone.

**Texture:** deliberate pace; reads everything; hovers unfamiliar terms; moderate–long dwell
on the price wall; one quiet idle gap when she's weighing "call later vs finish now". No
rushing, no anger.

### CHURN (her dominant path) — at S4 price wall
S1 Bei Arztbesuchen → S2 Ich selbst → S3 fill DOB+SV cleanly (low friction) → **S4**: hover
`augen_op` and `hilfsmittel` (i) tooltips (unfamiliar terms), `price_hover` on Optimal,
click **Opt.Plus** (`premium_click`) — sees "advisory required", which *irritates* her
("I wanted to do this online") → long dwell → one idle gap → **leave / close page**, reason
e.g. `will_call_advisor` or `think_about_it`.

### CHURN (secondary) — at S6 final price
…persists past S4 with **Optimal**, completes S6, sets health **yes** so final price ticks
up vs the estimate → small pause (`price_hover` / cancel hover) → **abandon**
`price_higher_than_estimate`.

### CONVERT (minority, real) — online done when reassured
S1/S2 confident → S3 clean → **S4** reads tariffs, hovers a term or two, notes Opt.Plus is
advisory-only, **picks Optimal** *because she wants it finished online* → S6 health **no**
(price holds) → **Abschließen**. Thought arc: "fair value, I understand it, just finish."

---

## 2. Franz — Online Affine (50 %)

**Identity:** 40, Vienna, digital-first, average income, no patience for friction. Insurance
= a product, not a relationship. **Never** wants an advisor. Leaves **silently** (signal is
in the click pattern, not words). Comparison-mind always on.

**Arrival context (1st thought):** was actively comparing offers (Durchblicker/Check24),
situation changed (considered freelance, bad public-care wait, friend said private "isn't
that expensive"). Intends to **finish in this one session**.

**Texture:** fast, decisive, short bursts; **rushed** mouse tone early; ~60 s focused
evaluation at S4; realistically opens a competitor tab → `tab_blur` then `tab_focus` back.
Strong negative reaction to "advisory required" and to any price that moves.

### CONVERT (happens when the price holds)
S1/S2 **fast** → S3 quick keystrokes → **S4**: compare, maybe `tab_blur`→`tab_focus`
(checking a competitor), avoid Opt.Plus/Premium (advisory wall), **pick Optimal** → S6 fill
honestly, health **no** → final price ≈ estimate (no surprise) → **convert**.

### CHURN (his canonical exit) — final-price jump at S6
…reaches S6 with Optimal, health **yes** → final price jumps vs the €68 estimate → he feels
mildly deceived (page never warned him) → `price_hover` / hover cancel→continue → **abandon**
`final_price_differs`. Quick, decisive close — **no** long idle.

### CHURN (secondary) — advisory wall at S4
At S4 clicks **Premium** (`premium_click`), hits "advisory required" = a wall → brief
`tab_blur` → **leave** `advisory_required_wall`. "Why am I being pushed into something I
didn't ask for?"

---

## 3. Peter — Service Affine (20 %)

**Identity:** 35, urban-edge/rural, service job, tight income, low bandwidth for admin.
Overwhelmed by tables/jargon. Wants **one recommendation**, not a menu. His ideal outcome is
a **warm handoff to a person** (callback) — for UNIQA that *is* a conversion, but in the
static funnel (no coach yet) he mostly **leaves early**.

**Arrival context (1st thought):** did **not** plan to be here — something prompted it (a
hospital bill, worried family member, colleague). Clicked a Google ad. Half-expects to just
call instead.

**Texture:** hesitant, slow, skims (misses things), dwells on complex pages **without
choosing**, back-navigates, hovers many things without clicking, anxious before entering
personal data. Easily paralyzed by more choice.

### CHURN (dominant, and EARLIER than the others)
S1 hesitates — hovers **both** cards, unsure which applies, maybe checks both → S3 anxious
dwell entering DOB/SV ("am I committing to something?") → **S4**: scans, too many numbers,
hovers several (i) tooltips with no selection, **looks for a "recommended for you" tag —
there's none**, long idle + a back-nav → **leave** `overwhelmed_complexity` / `will_call`.
He often bounces at **S3 or S4**, before Judith/Franz would.

### CONVERT (rare but real — simple-product Peter)
Determined / pre-decided socially (someone told him which to get): S1 Bei Arztbesuchen →
S2 Ich selbst → S3 slow but completes → **S4 picks Start** (cheapest, clearest — tight
income, doesn't over-analyze) → S6 health **no** → push through → **convert**. Slow, lots of
dwell, but finishes.

---

## 4. Recording spread (for the HUMAN reference set only)

Record **~30–40 per persona**, varying the **arrival context** each time (different trigger,
mood, time pressure). Let the outcome follow the archetype's natural tendency — don't force a
ratio, but the set should *look like* the archetype:

| Persona | mostly… | converts… | early exits (S3) |
|---|---|---|---|
| Judith | churns at **S4** (price wall, advisor lean); some at S6 | occasionally (finishes Optimal) | rare |
| Franz | churns at **S6** (price jump); some advisory-wall at S4 | **more often** than the others, when price holds | almost never |
| Peter | churns **early (S3/S4)**, overwhelmed | rarely (Start, pre-decided) | yes, sometimes |

Expect roughly **1–4 conversions per 35 sessions** for each persona (so each persona converts
at least once). These counts are a *sanity check on realism* of the human set — **not** a quota
to type into any agent. The anchors we ultimately validate against (S4 ≈ 66 %, S6 ≈ 78 %,
overall ≈ 5.6 %) stay in `funnel.py`.

---

## 5. Anti-leakage rules (so targets stay emergent)

What the persona agent (and every training example) **may** see:
- The persona's identity, values, annoyances, channel preferences, behavioral mechanisms
  (where they slow down, what irritates them, how they pace).
- The **static world model**: the funnel steps, the action space per step, the structural
  rule that hospital/other-persons/Opt.Plus/Premium route to an advisor (= online abandon).
  This is UI mechanics, not a churn rate.

What it **must NOT** see:
- Any aggregate funnel statistic: `66 %`, `78 %`, `24 %`, `5.6 %`, `34 % survive`, the
  30/50/20 mix, or sentences like *"That is the 66 % drop-off… it includes you."*
- `funnel.py`'s `ABANDON_PROBS`, `PERSONA_WEIGHTS`, psyche hazard tables.

**Action (manual, not regex):** the agent system prompt is a **hand-scrubbed,
version-controlled file per persona** under `prompts/personas/{judith,franz,peter}.md`.
These are seeded from the briefings with the funnel-target sentences removed **by hand**
(e.g. *"That is the 66% drop-off… includes you"* → *"This first price screen is the moment
you most often quietly disengage"*; Peter's *"20% of traffic / most won't complete online"*
note removed). `persona_datagen.agent_persona_prompt(persona)` reads exactly these files —
no runtime regex, no auto-injected `personas.json` numbers. To change what the agent sees,
edit the file; the diff is the audit trail. A read-only test
(`test_agent_prompt_files_have_no_funnel_targets`) fails loudly if a rate sneaks back in.
Behavioural/channel priors (60% customer-service, 24% switch, 39% online) are intentionally
kept — they shape behaviour and are not churn/bounce/conversion targets.

---

## 6. The "thoughts" trace (agent-generated training signal)

Human logs carry behavior + timing (and a per-step mouse *tone*), but a clicking human can't
narrate — so **thoughts come from the LLM agent** and become the supervised reasoning signal
for the fine-tune.

Per-event `thought` = short first-person motivation/expectation. Rules added to the
whole-session prompt:
- **The first event's thought sets the context**: who is arriving, what triggered this visit,
  what they expect/want from the session ("colleague said UNIQA's private cover is cheap; I
  want a number before I commit").
- At **price reveals**, the thought must express **expectation vs reality** ("€68? bit more
  than I hoped, but ok" / "wait, €71 — the estimate said 68, that's annoying").
- Thoughts should reveal the **gap between stated and real reason** at abandon (Judith thinks
  "I'll think about it" but the real driver is the advisor lean).

These `thought` fields ride inside the session trace (`Event.thought`, already in
`contracts.py` and emitted by `parse_session`), so the training trace = events **+** reasoning.

---

## 7. Pipeline (where this fits)

```
[1] HUMAN capture (this doc)         → _local/captures/<persona>/*.json   (behavior+timing truth)
[2] AGENT gen 30–40/persona          → python -m persona.persona_datagen    (LLM teacher, target-free prompt)
[3] COMPARE human vs agent           → python -m uniqa.compare … + per-persona stat/pattern diff
        → adjust PROMPT (not targets) until conformance + human-like patterns hold
[4] FINAL datasets vs static widget  → with thoughts; 1st thought = context
[5] FINE-TUNE small model (Leonardo) → local persona models (per docs/PIPELINE_PLAN.md §5.1)
[6] EVAL local models                → same stat properties vs funnel.py anchors
        → improve training set until local ≈ frontier-LLM behavior
[7] introduce COACH widget           → 100s of sessions → fast autoresearch iterations
```

Gates: step 3 closes when `ε_teacher_vs_psyche ≤ ~0.05` **and** human/agent dwell + event-mix
distributions overlap; step 6 closes when the local model reproduces the same per-persona
convert/bounce profile the frontier teacher does.
