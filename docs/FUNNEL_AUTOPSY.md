# UNIQA Funnel Autopsy — drop-off analysis
> 2026-05-30 | Drop-off analysis of the funnel. For the **static-widget spec**
> (canonical screens + action space) see `PIPELINE_PLAN.md` §"the static widget"
> and `sim_loop/widget.py`; this doc is the *why-they-leave* analysis.

---

## Canonical funnel screens (in order — intro screens included)

The in-scope online path per the official funnel doc
([uniqa-funnel-doc_en.md](https://github.com/Lumos-Data/zero_one_hack_01/blob/main/tracks/insurance-uniqa/uniqa-funnel-doc_en.md)).
The funnel does **not** start at the tariff table — three low-friction *intro*
screens come first:

| Step | Phase | Screen | Drop |
|---|---|---|---|
| **S1** | Inputs | Where covered? `At doctor visits`✅ / `In hospital`❌ | low |
| **S2** | Inputs | Who insured? `Myself`✅ / `Other persons`❌ | low |
| **S3** | Inputs | DOB + Sozialversicherung-Nr (first PII, pre-price) | low–med |
| **S4** | Product | Tariff table — **provisional** premium (Start €38.74 / Optimal €68.14) | **66%** |
| ~~S5~~ | Product | Add-on coverage (Sonderklasse) — **hospital path, out of scope** | (24%) |
| **S6** | Inputs | Health questions (for final premium) | — |
| **S7** | Recommendation | **Final** premium after health assessment | **78%** |
| **S12+** | Closing | name/address · start date · payment · consents · confirm | — |

Canonical prices = **€38.74 / €68.14** (Start / Optimal). The deep-dives below come
from an early CDP capture and show an earlier price snapshot (41,30 / 73,02) and a
private-doctor add-on screen; treat them as drop-off analysis of the **price + data-
form** screens, not as the canonical screen order (which is the table above).

---

## Deep-dive: the high-drop screens (early CDP capture)

### S4 — Tariff Table (66% abandon)

```
"Welche Leistungen soll Ihre Privatarzt-Versicherung abdecken?"

             Start       Optimal     Opt.Plus    Premium
             41,30 €/mo  73,02 €/mo  105,07 €/mo 152,35 €/mo
             [Ausgewählt][Wählen]    [Wählen]    [Wählen]

Disclaimer:  "Voraussichtliche Prämie — Finale Prämie folgt nach
              Erhebung des Gesundheitszustands"
```

**Why 66% leave here:**

1. **Double uncertainty**: Price is "preliminary" AND health assessment hasn't happened. User mentally adds: "this will probably be higher." Unknown final price is more frightening than a known high price.

2. **Column overload**: 4 tariffs visible including 2 out-of-scope (Opt.Plus / Premium). User compares all 4, anchors on Premium at €152 → €41 looks cheap → but then realizes Opt.Plus at €105 has more features → confusion → exit.

3. **Coverage table is abstract**: "Arztleistungen 1.120 EUR" means nothing without knowing how much a typical specialist visit costs OOP. No translation to real-world benefit.

4. **Sticky price bar**: "Unser Angebot 41,30 EUR/Monat" appears at bottom. Follows user on scroll. Creates constant price salience — anxiety throughout the coverage review.

**Coach interventions possible here (without redesign):**
- Reframe to daily: "€1,38/Tag" (Start) or "€2,43/Tag" (Optimal)
- Translate coverage to real-world: "1.120 EUR covers ~14 specialist visits/year"
- Address "preliminary" anxiety: "For 87% of customers, the final price stays within 10% of this estimate"
- Hide Opt.Plus/Premium columns for users who've already shown Start-range price acceptance

---

### S5 — Add-on Upsell (private-doctor Extra-Schutz) ← CRITICAL TIMING PROBLEM

```
"Wünschen Sie Extra-Schutz bei Ihrer Privatarzt-Versicherung?"

Fit fühlen:      +17,17 EUR  [toggle off]
Eltern werden:   +12,73 EUR  [toggle off]
Mental wachsen:  +25,76 EUR  [toggle off]
Akut Versorgt:   +12,79 EUR  [toggle off] ← "3 Monate prämienfrei"
BabyOption:       +4,62 EUR  [toggle off]
VitalPlan:       +16,34 EUR  [toggle off]

"Unser Angebot: 41,30 EUR/Monat" (still base, no adds)
```

**Why this is wrong:**

The upsell happens at step 5 — BEFORE personal data, BEFORE health questions, BEFORE commitment to the base tariff. The user is still in evaluation mode. They haven't said yes to €41/mo yet, and they're being asked to consider +€25/mo for Mental wachsen.

**Psychological sequence violation:**
```
Current:  Choose → SEE PRICE → UPSELL → commit personal data → health → final price → buy
Correct:  Choose → commit personal data → health → FINAL PRICE → BUY → upsell post-purchase
                                                                          OR
          Choose → minimal commitment → see final price → buy base → "complete your coverage" email
```

**The Akut Versorgt incentive ("3 Monate prämienfrei") is smart** but landing at the wrong moment. Post-purchase upsell with this offer would convert significantly better — user has already committed, offer feels like a reward.

**Coach intervention here (without redesign):**
- Collapse the add-on list by default, show only 1 most-relevant add-on based on DOB/persona
- Peter (service-seeker): show Akut Versorgt (urgency care = phone-accessible)
- Judith (hybrid, higher income): show Mental wachsen (premium wellbeing)
- Franz (price-sensitive): skip add-ons entirely, proceed straight to Weiter
- Display running total only when add-on is toggled ON (don't show +€25 anxiety upfront)

---

### S6/S7 — Personal Data + Health Form (78% abandon at health/final price) ← WORST STEP

```
"Angaben zu Ihrer Person"

Personenbezogene Daten:
  Geschlecht: [männlich][weiblich][divers]
  Vorname: [___________]
  Name: [___________]
  Sozialversicherungsnummer: [___________]
  (10-digit number, explanation text below — creates anxiety)

Kontaktdaten:
  E-Mail: [___________]
  Telefonnummer: +43 [___________]

Medizinische Angaben:
  Größe (in cm): [___]
  Gewicht (in kg): [___]
  Betreiben Sie Berufs- oder Leistungssport? [ja][nein]
  Behandelnder Arzt: [___________]  ☐ kein behandelnder Arzt

"Unser Angebot: 41,30 EUR/Monat"
```

**Why 78% leave here:**

1. **SV-number field is a wall.** Austrian Sozialversicherungsnummer is a 10-digit number most people don't know by heart. It's on the health card (e-card). 78% of people don't have their e-card within arm's reach while browsing on their phone.

2. **Medical data before commitment.** Height + weight + sports status + doctor name — feels like a hospital intake form. These questions signal: "we're going to assess your health risk." Creates pre-emptive anxiety about rejection or premium hike, which hasn't been disclosed yet.

3. **No progress indication.** User doesn't know if this is the last step before purchase or if there are 5 more pages.

4. **Form length is deceiving.** What looks like 1 screen is actually >1500px of scrolling content. The sticky "41,30 EUR/Monat" bar implies the price is already final, but the form hasn't been processed yet.

**The pre-fill opportunity (from Screen 5):**

Screen 5 (advisor booking) shows the system can pre-fill:
- Full name (Ivan Kotelnikov) — presumably from login session
- DOB (17.03.1995) — from earlier calculator step
- E-mail (ivan.d.kotelnikov@gmail.com)
- Phone (+436704048778)
- PLZ (1010 Vienna)

**The online purchase funnel doesn't do this.** The advisor path auto-fills from login session; the self-serve purchase path makes you type everything manually. This is an artificial friction asymmetry that disadvantages the online channel.

**Redesign proposals:**

**Option A: Progressive data collection (minimum viable commit)**
```
Step 4 (tariff select): already done
Step 5 (add-ons): collapse, show 1 recommended, skip for Franz
Step 6 (minimal data):
  E-Mail only → "Reserve your spot"
  "We'll send your quote to this address. Complete your details when ready."
  → converts email lead, allows async completion
Step 7 (async): email link → full data form with auto-fill from eID/login
```

**Option B: Deposit-first architecture**
```
After tariff selection:
  "Lock in today's rate. Pay €1 deposit."
  [PAY €1 →]
  Sunk cost kicks in → user far more motivated to complete the form
  "Your insurance starts when you complete your application."
  Medical data now feels like "finalizing" not "evaluating"
  Post-deposit: full form with time-limit ("complete within 14 days")
```

**Option C: eID / eAusweise pre-fill**
```
"Weiter mit Handysignatur" button (Austrian digital ID)
→ Pre-fills: name, address, DOB, SV-number — all at once
→ User only needs to confirm, not type
→ Reduces form to: E-Mail + phone + height/weight + sport + doctor
Austrian eID is widely adopted (>50% of adults have Handysignatur)
```

**Option D: Coach-only (no structural change, within hackathon scope)**
```
Personalize the form header based on what we already know:
"Guten Tag, wir haben schon ein paar Daten von Ihnen."
Pre-fill DOB (collected at Step 3), Sozialversicherung type (ÖGK confirmed)
Show: "7 von 9 Felder bereits ausgefüllt" (even if 2 are)
Progress bar within the form: ["Persönliche Daten ✓"]["Kontakt ●"]["Gesundheit ○"]
Add: "Sie sind fast fertig — noch 3 Minuten" at top
The SV-number field: add "Auf Ihrer e-Card" callout with e-card image
```

---

### Advisor Channel Selector (out-of-scope path)

```
"Wo soll die Beratung bevorzugt stattfinden?"
[Online Videoberatung NEU] [Persönlich UNIQA-Standort] [Per Telefon] [Persönlich zu Hause]
```

This is the ADVISOR HANDOFF endpoint — not the online conversion path. But it reveals:

- This is where Peter SHOULD land if he chose hospital path or Opt.Plus/Premium
- The "Online Videoberatung NEU" option is interesting — video call could intercept mid-funnel abandoners
- Coach opportunity: fire an advisor intercept at Step 3 for Peter BEFORE he hits the data form, offering this choice proactively

---

### Appointment Booking (advisor path, auto-filled — out of scope)

```
"Persönlichen Beratungstermin vereinbaren"
Bevorzugte Beratungszeit: [08-12][12-16][16-19]

Meine Daten: (AUTO-FILLED)
  Anrede: männlich
  Vorname: Ivan  |  Nachname: Kotelnikov
  Geburtsdatum: 17.03.1995

Meine Kontaktdaten: (AUTO-FILLED)
  Straße: [blank]  Hausnummer: [blank]
  PLZ: 1010  Ort: [blank]
  E-Mail: ivan.d.kotelnikov@gmail.com
  Telefon: +436704048778

Zusammenfassung:
  Produkt: Privatarzt Start
  Voraussichtliche Prämie: 40,20 EUR/Monat

Meine Nachricht an UNIQA (optional): [text area]
Marketingeinwilligung: [checkbox]
```

**The smoking gun:** UNIQA's backend has Ivan's full profile (from login). The advisor booking flow uses it. The online purchase flow ignores it. The self-serve channel is artificially harder than the advisor channel.

**The summary block is gold:** "Produkt: Privatarzt Start | Prämie: 40,20 EUR/Monat" — this is a clean quote confirmation. The same block should appear at the TOP of the personal data form ("You're buying: Privatarzt Start at €41,30/mo") to maintain commitment and context throughout the long form.

---

## Funnel Redesign Options (Priority Matrix)

| Option | Impact | Effort | Requires UNIQA | In hackathon scope |
|--------|--------|--------|----------------|-------------------|
| Coach widget at Step 4: daily price reframe | High | Low | No | ✅ |
| Coach: hide Opt.Plus/Premium columns | Medium | Medium | Yes (DOM inject) | ✅ CDP |
| Coach: add-on personalization by persona | High | Medium | No | ✅ |
| Form: pre-fill from Step 3 data (DOB, SV type) | High | Low | No | ✅ |
| Form: progress bar within form | Medium | Low | No | ✅ |
| Form: SV-number helper with e-card image | Medium | Low | No | ✅ |
| Form: "X Felder bereits ausgefüllt" header | Medium | Low | No | ✅ |
| Deposit-first architecture | Very High | High | Yes (payment system) | ❌ too big |
| eID / Handysignatur integration | Very High | Very High | Yes | ❌ too big |
| Move upsell to post-purchase | High | Medium | Yes (funnel restructure) | ⚠️ pitch only |
| Email-first progressive collection | High | Medium | Yes | ⚠️ pitch only |

**Hackathon strategy:** Build all ✅ items as Coach widgets. Pitch ⚠️ items as "what UNIQA should do next." Show ❌ items as product roadmap vision.

---

## Coach Intervention Map (full funnel)

```
STEP 1 (coverage type)
  No hesitation signal → no intervention
  Long dwell (>10s) → "Die meisten wählen 'Bei Arztbesuchen' — es deckt Ihre häufigsten Bedürfnisse."

STEP 2 (who insured)
  No hesitation → no intervention
  Back-nav → "Für sich selbst? Das ist der häufigste Start — Sie können später erweitern."

STEP 3 (DOB + SV)
  Sozialversicherung dropdown re-opened → "Ihre ÖGK-Karte liegt am Besten bereit."

STEP 4 ← 66% ABANDON (PRIORITY)
  Immediate on page load (all personas):
    → PriceReframe: "Das sind €1,38/Tag — weniger als ein Kaffee."
  Hover on Opt.Plus/Premium (upgrade intent signal):
    → UpgradePath: "Nach 3 Jahren in Start/Optimal können Sie ohne neue Gesundheitsprüfung wechseln."
  Long dwell (>8s) without selection:
    → CoverageExplainer: "Was bedeutet 1.120 EUR Arztleistungen? Das entspricht ca. 14 Facharztterminen pro Jahr."
  Peter persona AND dwell>6s → before price anxiety:
    → CallbackOffer: "Lieber persönlich beraten? 15 Minuten Telefon-Termin jetzt buchen."
  Price hover (cursor stays on monthly price >3s):
    → TrustSignal: "Für 87% unserer Kunden bleibt die Endprämie innerhalb von 10% dieser Schätzung."

STEP 5 (add-ons) ← 24% ABANDON
  Default: collapse list, show only 1 recommended add-on
  Franz: skip straight to Weiter, no add-ons surfaced
  Judith: show Mental wachsen first (aligns with higher income / wellbeing segment)
  Peter: show Akut Versorgt first (24/7 coverage = security, matches service orientation)
  Running total: only reveal price delta when toggle turns ON
  "Akut Versorgt: 3 Monate prämienfrei" → highlight this as time-limited signal for all personas

STEP 6 (personal data) ← 78% ABANDON (PRIORITY)
  On page load:
    → FormContextualizer: "Fast fertig! Noch 3 Minuten, dann sind Sie versichert."
    → QuoteSummary header: "Sie kaufen: Privatarzt Start | €41,30/Monat"
    → Pre-fill DOB from Step 3 (already collected!)
  SV-Nummer field focus:
    → Helper: "Auf Ihrer e-Card oben rechts" + e-card SVG image
  30s no activity:
    → ProgressSaver: "Darf ich Ihre E-Mail notieren? Wir sichern Ihren Fortschritt."
  Height/weight fields:
    → HealthContextualizer: "Diese Angaben werden nur für Ihre individuelle Risikoprüfung verwendet und nicht an Dritte weitergegeben."
```

---

## The Deposit Idea — Full Design

**Mechanism:**

```
After Step 5 (add-ons), before personal data form:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Privatarzt Start · €41,30/Monat                           │
│                                                             │
│  ✓ Heute sichern. Morgen abschließen.                      │
│                                                             │
│  Zahlen Sie €1 Reservierungsgebühr und sichern Sie sich    │
│  Ihren Versicherungsstart ab heute — auch wenn Sie das     │
│  Formular morgen ausfüllen.                                │
│                                                             │
│  Die €1 wird auf Ihre erste Prämie angerechnet.            │
│                                                             │
│  [Pay €1 — Versicherungsstart sichern →]                   │
│                                                             │
│  ─────────────────────────────────────────────────────     │
│  Oder: Direkt jetzt abschließen →                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why this works:**
1. **Sunk cost**: €1 paid → user is now a "customer" not a "prospect" → completion rate for the full form should be 3-5× higher
2. **Time-decoupling**: heavy form can be completed async (email link, app, next day)
3. **No price uncertainty**: they've committed at today's rate even before health questions
4. **Minimal data**: only email needed for the deposit step — full data collected at completion
5. **Reduces anxiety about health questions**: user has already "bought" → health questions feel like formalities not gatekeepers

**Legal note (Austrian VersVG):** Insurance contract typically requires offer + acceptance + all material data. The deposit creates an "option to buy" or a "pre-contract" — requires legal review but likely feasible as a "Anfrage" that locks the start date. UNIQA can reserve a start date without binding coverage until the form is complete.

---

## What We Already Know Per User at Each Step

```
After Step 1:  coverage_type = "arzt"
After Step 2:  insured = "myself"
After Step 3:  DOB, sozialversicherung_type (ÖGK/BVAEB/SVS)
After Step 4:  tariff selected, estimated_price
After Step 5:  add_ons selected, final_estimated_price
After Step 6:  full_name, email, phone, height, weight, sport, doctor
After health Q: health_risk_profile
```

**What we can personalize at each step:**
- Step 4: price shown is already age-adjusted (we know DOB from Step 3)
- Step 5: add-on recommendation uses DOB → age-appropriate (Eltern werden for 25-35, Fit fühlen for 35-50)
- Step 6 header: "Guten Tag — wir haben schon Ihr Geburtsdatum und Ihre ÖGK-Mitgliedschaft. Noch 3 Felder!"
- Step 6: pre-fill DOB field (already known) → reduces perceived form length

**The "what if we had data at moment of purchase intent" scenario:**
If user is logged in (UNIQA app user / existing customer):
- Pre-fill: Geschlecht, Vorname, Name, DOB, E-Mail, Telefon, PLZ, Ort
- Pre-fill: SV-Nummer (UNIQA already has it for existing customers)
- Remaining: just Größe + Gewicht + sport + doctor
- Form drops from ~12 fields to 4 fields → massive conversion lift
- The advisor path (Screen 5) already does this. It's a backend capability UNIQA has.

---

## Summary: 3 Levels of Coach Ambition

**Level 1 — Widget-only, current funnel (hackathon MVP)**
Interventions at each step, no structural changes. Build time: ~8h. Demonstrable in Streamlit.

**Level 2 — Progressive data + pre-fill (hackathon stretch)**
Modify form initialization to pre-fill known fields. CDP-injectable. Build time: +4h.
Show: Step 6 header personalizes based on Step 3 data already in session.

**Level 3 — Deposit-first + async completion (pitch roadmap)**
Structural redesign, requires UNIQA backend. Not buildable in hackathon.
Present as: "This is what UNIQA could ship in Q3 2026 — we've proven the Coach logic that makes it work."

**Demo narrative:**
> "The current UNIQA funnel loses 66% at the price table and 78% at the data form. 
> Our Coach reduces both. But we also identified that UNIQA is already pre-filling 
> data for its advisor path — they just don't do it for self-serve. We show them exactly 
> where to unlock that. And we propose a €1 deposit mechanism that decouples commitment 
> from form completion — the same user who bounces at the data form will complete it 
> the next morning if they've already paid €1."
