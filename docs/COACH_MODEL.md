# The Coach model — detection, decision & intervention

> **Design vs implemented.** `sim_loop/coach.py` implements the live coach: the detection→decision
> `COACH_SYSTEM` prompt + the 32-effector `EFFECTOR_LIBRARY` (json-render). The richer signal set
> and the **cohort-relative baseline** (§3.4) below are the full *design* — not all of it is wired
> into sim_loop yet (the autoresearch loop, `sim_loop/autoresearch.py`, is how the prompt earns each rule).

The Coach watches the user's **event trace over time** and decides whether to show ONE
helpful widget, and which. It reasons from **principles, not rigid rules**. Live policy:
the `COACH_SYSTEM` prompt + `EFFECTOR_LIBRARY` in `sim_loop/coach.py`. This doc is the *why* — model, signals,
hypotheses, and the per-screen intervention design.

> Merges the former COACH_MODEL · SIGNALS_AND_DETECTION · HYPOTHESES · COACH_HYPOTHESES ·
> INTERVENTION_DESIGN into one source of truth.

---

## 1. Conversion target is PERSONA-DEPENDENT (read this first)

Online purchase is **not** always the goal. Per UNIQA's per-segment definition the Coach
optimizes the RIGHT outcome — the same "bounce risk" routes to a different *win*:

| Persona | Conversion target |
|---|---|
| **Judith** (Rising Hybrid, 30%) | online purchase **OR** smooth advisor handoff — **both count** (don't force online for the ~81% who won't) |
| **Franz** (Online Affine, 50%) | **online purchase only** — advisor/offline handoff = FAILURE (Medienbruch is his pain) |
| **Peter** (Service Affine, 20%) | **qualified service contact** (callback booked / phone / WhatsApp) — online purchase is NOT the target |

## 2. Funnel ground truth + step roles

Official churn model (UNIQA slide, conditional on reaching each step) — sample **1000 → 333 → 253 → 56**:

- **S4** first price display: 1000→333 = **66.7%** drop
- **S5** add-on selection: 333→253 = **24.0%** drop
- **S6** final price (after personal+health): 253→56 = **77.9%** drop
- conversion = 56/1000 = **5.6%**

Traffic: ~80% from search; 70%+ between 9–20h. ROPO (look online, buy via advisor) is real and unmeasurable.

| Steps | Role | Coach stance |
|---|---|---|
| **S1 coverage · S2 insured** | **persona DETECTION** | observe only (which card, dwell, hover) → infer persona; ~no churn |
| **S3 personal info** | intervention | big-form scare → pre-emptive explainer |
| **S4 first price** | intervention | price reframe / package nuance / Premium-online clarifier |
| **S5 add-on** | intervention | "skip is fine" reassurance / value framing |
| **S6 final price + big form** | intervention | explain price jump, form rationale, graceful options |

**Eval policy.** PRIMARY gate = survival-weighted **population aggregate vs UNIQA** (S4 66.7% ·
S5 24% · S6 77.9% · conversion 5.6%), conditional. Per-persona `ABANDON_PROBS` are OUR
decomposition → SECONDARY diagnostic only (their *shape* must stay consistent with the segment
descriptions, e.g. Judith > Franz at S4).

---

## 3. Signals & detection

### 3.1 Traffic source — a session prior AND a persona signal
~80% of calculator traffic is Paid + Organic Search; entry = `uniqa.at/rechner/krankenversicherung`.
Source correlates with intent and persona (hypothesis layer — inferred from per-segment channel prefs):

| Source | Intent | Persona prior |
|---|---|---|
| **paid_search** (intent keyword) | high — actively shopping | Franz↑, Judith↑ |
| **organic_search** | medium-high — info gathering | Judith / Franz |
| **price-comparison referrer** (Durchblicker/Check24) | high, comparison-mode | **Franz** (compares 82–94%) |
| **display / social ad** | low — interrupted, wasn't looking | **Peter** (passive) |
| **direct / portal** | warm — existing customer | Judith (portal 54%) |

Plus **device** (Peter 65% mobile; Franz/Judith desktop-lean), **time-of-day**, **returning vs new** →
seed the persona prior *before any click*.

### 3.2 Atomic events (trace vocabulary)
Base (`sim_loop/widget.py` event vocabulary): `step_enter, mouse_move, hover, pause, idle, scroll,
field_focus/blur/edit/invalid, keystroke, tap, select, dropdown_open, tooltip_open,
validation_error, price_reveal, price_hover, tariff_click, premium_click, nav_back, session_gap,
tab_blur, tab_focus, cancel_hover, submit, abandon, convert, widget_shown/cta/dismiss`.

Rich repertoire (added): `exit_intent` (cursor darts to edge → about to leave; strongest last-chance
trigger) · `text_select` (reading hard / jargon) · `copy` (about to google a term / leave to compare)
· `external_nav` · `compare_return` (value=sec away: short=compared, long=forgot) · `slow_mouse`
(deliberation) · `rage_click`/`repeat_click` (frustration) · `scroll_up`/`reread` (confusion) ·
`field_clear` (doubt).

### 3.3 Derived signals (scored OVER TIME, not single events)
**Hesitation/friction:** back_nav_count, field_reedit_count, validation_error_count, field_clear_count,
longest_idle_sec, time_to_first_action, `hesitation` (0..1, persona-emitted).
**Info-seeking/confusion:** tooltip_open_count, hover_count, scroll_reversals, text_select_count,
term_copy_count, slow_mouse_ratio.
**Comparison/leaving:** tab_away_count, tab_away_total_sec, return_time, external_nav_count, exit_intent_count.
**Price reaction:** price_hover_count, cancel_hover_count, dwell-after-price_reveal.
**Effort/momentum:** keystroke_total, taps_total, step_dwell_sec, step_revisit_count, momentum, total_session_sec.

### 3.4 Cohort baseline & RELATIVE signals (anomaly detection, not fixed thresholds)
"Fast" is meaningless absolutely — only relative to peers. `sim_loop/coach.py` computes a **cohort
baseline** (per step×metric mean/std/p50/p90 over many sessions, cohort = e.g. this week × device);
each live session is scored as **z-divergence** and the Coach acts on **OUTLIERS** (`|z| ≥ ~2`):
- `dwell_sec` z ≈ −2.3 early → much faster than peers → decisive (Franz) or skimming.
- `back_nav_n`/`cancel_hover_n`/`validation_err_n` z high → anomalously stuck → friction.
- `tab_away_sec` z high → left far longer than peers → forgot (vs short = compared).

`baseline.build_baseline(logs)` + `baseline.divergence(session, baseline)` → `relative_signals` fed
to the coach. Baselines are cohort/time-window-scoped → recompute per window.

### 3.5 Per-persona fingerprints + early detection
- **Judith** (research→advisor): deliberate; hover/tooltip; price-performance focus; S4 `nav_back` then
  graceful exit. 81% won't finish online → recover via advisor handoff. Desktop, portal-registered.
- **Franz** (fast online comparer): mechanical/fast input, low dwell; early tab_away/external_nav +
  compare_return; hates friction & advisor; primary drop at **S6** when final > expected. Desktop.
- **Peter** (passive, overwhelmed, mobile): slow_mouse, high time_to_first_action, early
  validation_error/field_reedit on S3; exits early; 65% mobile. Service-affine.

**Early detection (S1–S2)** combines traffic-source prior × device/time × early behavior:
fast decisive + desktop + paid/comparison → **Franz**; deliberate hover/tooltip + portal → **Judith**;
slow + landing-hesitation + mobile + display/social → **Peter**. Output a running `persona_belief`
distribution → commit ~by S3/S4 and **tailor the widget set** ("do not unify").

---

## 4. The 5-step decision workflow (run on EVERY event increment)

1. **Persona belief** — confidence distribution over {judith, franz, peter}, updated each event
   (traffic_source, device/time, speed of early steps, accumulating micro-signals). Sharpen over time;
   commit ~by S3/S4. As confidence rises, the option set **narrows** to that persona's playbook.
2. **Pains & frustration** — infer pains from micro-signals (price-shock freeze/exit-intent, form
   overwhelm, term confusion via text_select/copy, compare-intent via tab-away, forgot-a-field) +
   frustration 0..1, with per-pain confidence.
3. **Dropout likelihood → intervention temperature** — estimate P(bounce this step). Willingness to
   spend a widget = a **temperature** rising with `dropout_likelihood × belief_confidence`. Low temp →
   WAIT; high temp + clearly matching widget → act. `exit_intent` spikes temp. Respect ≤3 widgets/session.
4. **Widget match** — pick the intervention whose *purpose* addresses the detected pain AND serves
   THIS persona's conversion target, on THIS step, for THIS device. Least-intrusive pattern that works;
   escalate intrusiveness only with temperature.
5. **Feedback** — adapt to prior-widget feedback: `widget_dismiss` → back off / don't repeat;
   `widget_dislike` → change tactic; `widget_cta`/`widget_like` → read them right, may follow up once.

**Output (strict JSON):** `persona_belief`, `belief_confidence`, `detected_pains`, `frustration`,
`dropout_likelihood`, `intervention_temperature`, `decide` (wait|intervene), and `intervention`
{id, conversion_target, fe_pattern, device_variant, copy, reasoning, hypotheses} or null. WAIT is default.

---

## 5. Hypotheses registry (H1…) — priors to TEST, not rules

`sim_loop/coach.py` cites these by number; confirmed/refuted hypotheses recalibrate both the coach
policy and the persona model.

**Persona detection.** H1 fast early steps → Franz (don't interrupt early; save budget for S6).
H2 slow+overwhelm → Peter (early warm handoff; simplify). H3 deliberate research → Judith (term help +
comparison → graceful advisor). H4 traffic source is a prior.

**Conversion-likelihood / momentum.** H5 fast-fill → offer `jump_to_pricing`. H6 confident tariff pick →
high P(convert), don't over-intervene. H7 momentum is signal (steady = leave alone; stall/regress = act).

**Pains → moves.** H8 price-table shock (freeze/cancel_hover/exit_intent) → persona-routed reframe /
strip tariffs / comparison. H9 leave-to-compare (tab_away→fast compare_return) → `comparison_table`; long
away → re-orient. H10 term confusion (text_select/copy/scroll_up) → `coverage_explain`/`faq_cards`.
H11 forgot a field → `field_defer`. H12 big-form scare → `form_explainer`/`form_simplify`/`bucket_input`.
H13 premium dead end (premium_click→nav_back) → "Optimal is fully online" clarifier.

**Price transparency.** H14 pre-indicate price-affecting fields (`price_preview`); H14b if final >
estimate, always explain WHY (`value_justification`/`health_explain`). H15 bucket not exact (`bucket_input`).

**Channel / capture.** H16 Peter wants a human → early `callback_offer`/`whatsapp_bot`. **H16b Peter's
PERFECT move = `contact_handoff`** ("don't fill this — leave email/phone, we take it from here"; skips ALL
forms; IS his conversion). H17 mobile → phone capture (bottom sheet). H18 `id_austria_login` autofill.
H19 persist partial forms + `save_progress`/resume.

**S5 add-on.** H20 suggest skipping the add-on (`addon_skip_ok`) — protects conversion (24% drop).

**Feedback.** H21 tooltip hover = it helped → surface a Like button; close = negative.
`widget_like/dislike/dismiss` feed back (dismiss → back off; dislike → change tactic; like/cta → follow up once).

---

## 6. Intervention design (per screen × frontend pattern)

### 6.1 Frontend pattern library (HOW an intervention renders)
| Pattern | Best for | Intrusiveness |
|---|---|---|
| Inline banner | contextual explain, reassurance | low |
| Anchored popover / tooltip | term explain, field hint | very low |
| Inline expand (accordion) | package nuance, "what differs" | low |
| Coachmark / spotlight | guide the eye, preselect hint | medium |
| Side drawer | comparison table, running assistant | medium |
| Bottom sheet (mobile-first → Peter) | callback/WhatsApp, simplify | medium |
| Toast / snackbar | micro-reassurance, "saved" | very low |
| Sticky action bar | save-progress, callback, price chip | low |
| Price-reframe chip (€/day) | price reframe at S4/S6 | very low |
| Exit-intent overlay | last-chance (save/advisor/reframe) | high (rare) |
| Chat bubble / avatar | Peter guidance, open-question answers | low–medium |
| Progress ribbon ("2 fields · ~40s") | big-form pre-emptive | very low |

**Rule:** intrusiveness scales with churn-imminence — chips/banners early, exit-intent overlay only when
`exit_intent` fires. Never exceed 3 widgets/session.

### 6.2 Per-screen matrix (trigger → intervention → FE pattern → persona)

**S3 — personal info form** *(trust barrier, before any price)*
| Trigger | Intervention | FE pattern | Persona |
|---|---|---|---|
| landing hesitation≥.5 / high time-to-first-action | **form_explainer** ("why + ~3 fields, ~40s") | progress ribbon | all (esp. Peter) |
| field_reedit/validation_error on SV number | **form_helper** ("SV-Nr is top-right on your e-card") | anchored popover | all |
| early overwhelm (slow, multi back-nav, mobile) | **callback_offer / whatsapp_bot** | bottom sheet / chat bubble | **Peter** |
| copy/text_select of a term | **term explain** | anchored popover | Judith/Peter |

**S4 — first price / tariff table** *(66% drop; Judith's wall)*
| Trigger | Intervention | FE pattern | Persona |
|---|---|---|---|
| dwell-after-price_reveal + cancel_hover (shock) | **price_reframe** (€/day) | price-reframe chip | Judith/Franz |
| tab_away→compare_return (short) / external_nav | **market_comparison** ("Optimal < 80% of comparable") | side drawer | **Judith/Franz** |
| tariff hover/click w/o select + tooltip opens | **package_nuance** ("Start vs Optimal — 3 differences") | inline expand | all |
| premium_click then nav_back | **upgrade_explain** ("Optimal is fully online") | inline banner | Franz/Judith |
| copy/text_select of jargon | **coverage_explain / coverage_checker** | anchored popover | all |
| back-nav after "advisory required" + advisor prior | **advisor_handoff** (counts as conversion) | bottom sheet CTA | **Judith** |

**S5 — add-on selection** *(24% drop; upsell + cost bump)*
| Trigger | Intervention | FE pattern | Persona |
|---|---|---|---|
| dwell + running-price climbs + hesitation | **"skip is fine"** (add-ons optional, add later) | inline banner / toast | all (esp. Franz) |
| slow_mouse over many toggles | **quick_quiz / simplify** ("answer 2 questions") | bottom sheet | Peter |
| value doubt on an add-on | **value framing** (what each covers) | inline expand | Judith |

**S6 — personal+health form → FINAL PRICE** *(78% drop; Franz's wall)*
| Trigger | Intervention | FE pattern | Persona |
|---|---|---|---|
| big-form hesitation before filling | **form_explainer** ("last step · ~1 min · then binding price") | progress ribbon | all (esp. Peter) |
| final price > provisional (health loading) + stick | **value_justification / health_explain** (why it changed, still online) | inline banner + price chip | **Franz** |
| price-perf doubt at final | **comparable cheaper tariff** w/ feature compare | side drawer | Franz |
| clearly not finishing now | **save_progress** (email resume, NO advisor) | sticky bar | **Franz** |
| exit_intent on final price | **last-chance overlay** (reframe + save + persona route) | exit-intent overlay | persona-routed |
| height/weight friction (recalls unsure) | **form_helper** ("estimate is fine, update later") | anchored popover | all |

### 6.3 Form-tool widgets (FE functionality)
- `form_simplify` — show only required fields + split into small steps.
- `field_defer` — defer a field (SV/weight/health); **transparently flag if it moves the binding price**.
- `bucket_input` — replace an exact field with categories + **price impact per bucket** (e.g. height `<170 / ≥170`).

### 6.4 Device-aware rendering (design BOTH)
| | Desktop | Mobile |
|---|---|---|
| Pattern | anchored popover / side drawer / inline | **bottom sheet** / full-width card / sticky bar |
| Anchoring | hover/cursor-anchored, coachmarks | thumb-reachable, no hover; tap ≥44px |
| Exit-intent | cursor-to-edge overlay | scroll-up + app-switch / back-gesture heuristic |
| Forms | inline expand | `form_simplify`/`bucket_input` matter MORE |

Mobile leans Peter (65% mobile) → favour bottom-sheet callback/WhatsApp + simplify.

---

## 7. How it trains / improves

This is the policy the autoresearch loop optimizes against the persona simulator: the persona reacts to
and **assesses** each widget (helpful/engaging vs distracting); that signal + the realized outcome teach
the coach which widget works for which persona/pain/device. Demo: build on the replay demo (`sim_loop/replay/src`) + coach overlay (`sim_loop/replay/src/coachWidgets.tsx`); show the **persona belief + live signal trace**
in a side HUD so judges SEE detection happening; scripted autoplay per persona, each firing a different
pattern. The gate is **empirical** (`Δuplift > τ`); the formal Z3 certificate is deferred (deferred).
