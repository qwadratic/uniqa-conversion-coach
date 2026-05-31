# UNIQA Funnel Twin + Coach Overlay — Implementable Design

> Drives a 1-1 React twin of the real UNIQA calculator on top of `@json-render/react`,
> plus a separate **coach overlay** layer that sits on the *untouchable* static funnel
> (track rule). Both consume the same `contracts.ActivityLog` schema and feed the
> existing `sim_loop/replay/src/capture.js` recorder, so action-space parity with
> `sim_loop/widget.py` is preserved end-to-end.

---

## 0. Architectural split (mirrors `contracts.py`)

`contracts.py` defines two components: the **APP (immutable)** and the **COACH
(mutable)**. The twin mirrors that split as **two independent json-render runtimes**:

```
┌────────────────────────────────────────────────────────────────────────────┐
│  FUNNEL TWIN  (immutable: schema is data, never patched at runtime)         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  funnelSchema.json   ──▶  <JsonRenderer schema registry store/>      │  │
│  │  funnelRegistry      (Reef components, funnel actions)               │  │
│  │  funnelStore         (currentStep, formData, errors, touched, …)     │  │
│  └────────────────────────────────────────────────────────────────────────┘
│                              │ funnel actions emit ActivityLog events       │
│                              ▼                                              │
│                       captureBridge ──▶ Recorder ──▶ ActivityLog JSON       │
│                              ▲                                              │
│                              │ overlay events also feed the same Recorder   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  COACH OVERLAY  (mutable: decisions stream in at runtime)            │  │
│  │  coachRegistry       (CoachCard / Toast / Sheet / Tooltip / CTA …)   │  │
│  │  CoachLayer          renders each active CoachDecision via portal    │  │
│  │  effectorBridge      translates EffectorCommand → overlay OR a       │  │
│  │                      whitelisted funnelStore mutation (AUTOFILL,     │  │
│  │                      PRESELECT_TARIFF, …) via the immutable API      │  │
│  └──────────────────────────────────────────────────────────────────────┘
└────────────────────────────────────────────────────────────────────────────┘
```

Key invariants:

1. **The funnel schema is read-only at runtime.** The coach never edits
   `funnelSchema.json`. It can only render overlays (above) or call the
   `effectorBridge` API to request a whitelisted mutation (analog of
   `EffectorCommand.validate()` in `contracts.py`).
2. **Two stores, one capture log.** Funnel store and coach overlay store are
   separate. Both subscribe to the same `Recorder` so the captured `ActivityLog`
   is the single source of truth (matches `contracts.ActivityLog`).
3. **Every funnel action and every coach event goes through `captureBridge`** —
   no DOM-level shortcuts; everything is observable.

---

## 1. Component catalog (Reef → json-render registry)

The catalog declares **type names + prop types only** (no React code). The registry
binds these types to React implementations. One entry per Reef component captured
in `UNIQA_FUNNEL_SPEC.md`.

### 1.1 Catalog declaration — `sim_loop/replay/src`

```ts
// catalog.ts — typed UI contract (no implementations)
export const funnelCatalog = {
  components: {
    // ─── Layout / chrome ──────────────────────────────────────────────────
    Stack:           { props: { direction: "string", gap: "number" } },
    Heading:         { props: { text: "string", level: "number" } },
    Instruction:     { props: { text: "string" } },           // ur-body-75 muted line
    ProgressStepper: { props: { phases: "string[]", current: "number" } },
    StepLiveRegion:  { props: { heading: "string" } },        // aria-live announcer
    StickyOfferBar:  { props: { label: "string", price: "number" } },

    // ─── Choice cards (S1 multi-checkbox, S2 single-radio) ────────────────
    ChoiceCardGroup: { props: { mode: "string",               // "checkbox" | "radio"
                                 group: "string",              // state subtree key
                                 items: "object[]" } },        // [{id, title, icon, features[]}]
    ChoiceCard:      { props: { id: "string", title: "string", icon: "string",
                                 features: "string[]", checked: "boolean",
                                 mode: "string", onToggle: "action" } },

    // ─── Inputs (S3, S6) ──────────────────────────────────────────────────
    DatePicker:      { props: { label: "string", value: "string", error: "string",
                                 touched: "boolean", onChange: "action",
                                 onFocus: "action", onBlur: "action" } },
    SearchableSelect:{ props: { label: "string", value: "string", error: "string",
                                 touched: "boolean", placeholder: "string",
                                 options: "string[]", onOpen: "action",
                                 onPick: "action", onClose: "action",
                                 onFilter: "action" } },
    TextField:       { props: { label: "string", value: "string", error: "string",
                                 touched: "boolean", type: "string",
                                 onChange: "action", onBlur: "action" } },
    HelperTextError: { props: { text: "string" } },           // .helper-text.error

    // ─── S4 tariff table ──────────────────────────────────────────────────
    TariffTable:     { props: { tariffs: "object[]",          // [{id,name,price,online,maxYear}]
                                 selected: "string",
                                 onPickTariff: "action",
                                 onPremiumClick: "action" } },
    TariffColumn:    { props: { id: "string", name: "string", price: "number",
                                 online: "boolean", selected: "boolean",
                                 onClick: "action" } },
    CoverageRowTooltip: { props: { row: "string", body: "string",
                                    open: "boolean", onOpen: "action" } },

    // ─── S6 / S7 / closing ────────────────────────────────────────────────
    HealthForm:      { props: { values: "object", errors: "object",
                                 onField: "action", onSubmit: "action" } },
    FinalPrice:      { props: { price: "number", deltaFromProvisional: "number" } },
    ClosingForm:     { props: { values: "object", errors: "object",
                                 onField: "action", onSubmit: "action" } },

    // ─── Navigation ───────────────────────────────────────────────────────
    Button:          { props: { label: "string", variant: "string",   // "primary"|"secondary"
                                 disabled: "boolean", onClick: "action" } },
    NavRow:          { props: { showBack: "boolean",
                                 onBack: "action", onNext: "action",
                                 nextDisabled: "boolean" } },

    // ─── Per-step shell ───────────────────────────────────────────────────
    Step:            { props: { id: "string", index: "number", current: "number",
                                 phase: "string", heading: "string" } },
    Wizard:          { props: { current: "number" } },
  },

  actions: {
    // user-initiated funnel actions (mapped 1-1 to ActivityLog events in §3)
    toggleCoverage: { params: { id: "string" } },              // S1 checkbox
    pickInsured:    { params: { id: "string" } },              // S2 radio
    typeDob:        { params: { value: "string" } },           // S3 keystrokes
    openSvSelect:   { params: {} },
    pickSv:         { params: { value: "string" } },
    closeSvSelect:  { params: {} },
    filterSv:       { params: { query: "string" } },
    focusField:     { params: { field: "string" } },
    blurField:      { params: { field: "string" } },
    selectTariff:   { params: { id: "string" } },
    premiumClick:   { params: { id: "string" } },
    openTooltip:    { params: { row: "string" } },
    setField:       { params: { path: "string", value: "unknown" } },  // S6 generic
    submitHealth:   { params: {} },                             // S6 → triggers final price
    submitClosing:  { params: {} },                             // S12 → convert

    // navigation + lifecycle
    validateStep:   { params: { step: "string" } },             // runs per-step validators
    nextStep:       { params: {} },                             // gated by validation
    prevStep:       { params: {} },                             // "Zurück"
    cancelWizard:   { params: { reason: "string" } },           // S1 "Abbrechen"
    routeAdvisor:   { params: { reason: "string" } },           // out-of-scope terminal
    finishConvert:  { params: {} },                             // S12 terminal
  },
} as const;

export type FunnelCatalog = typeof funnelCatalog;
```

### 1.2 Component-to-Reef-element mapping (visual parity guide)

| Catalog type        | Real Reef element                | DOM hint                                  |
|---------------------|----------------------------------|-------------------------------------------|
| `ProgressStepper`   | UNIQA stepper                    | 4-phase bar (`Angaben › Produkt › …`)     |
| `ChoiceCard`        | UNIQA choice-card (S1, S2)       | Bordered card, icon + title + 3 ✓ bullets |
| `DatePicker`        | `UR-DATEPICKER`                  | `TT.MM.JJJJ` input + calendar button      |
| `SearchableSelect`  | `UR-SELECT` + CDK overlay        | Readonly trigger; portal list             |
| `HelperTextError`   | `.helper-text.ur-body-75.error`  | Red helper text + `UQ-UI-ICON unext-shared-error` |
| `TariffTable`       | 4-col tariff matrix (S4)         | Cards reveal price on select              |
| `Button[primary]`   | `Weiter`                         | UNIQA blue filled                         |
| `Button[secondary]` | `Abbrechen` / `Zurück`           | Outline                                   |

### 1.3 Implementation notes (registry — `sim_loop/replay/src`)

- Components import the **same `_local/styles.css` UNIQA palette** the current
  `sim_loop/replay/src/styles.css` already uses.
- `DatePicker` and `SearchableSelect` apply the Angular-style validation classes
  (`ng-invalid ng-touched` equivalent: `is-invalid is-touched` data attrs) so QA
  tooling and the coach can target them with selectors that mirror the real DOM.
- `StepLiveRegion` renders `<div role="status" aria-live="polite">Ein neuer
  Schritt wurde geladen: {heading}</div>` exactly as captured (§ recon).

---

## 2. Funnel as a json-render schema

### 2.1 State / data model (initial `funnelStore` state)

```ts
const initialState = {
  // navigation
  currentStepIndex: 0,                     // 0..4 maps to STEP_ORDER
  phase: "Angaben",                        // derived from current step
  terminal: null,                          // null | "convert" | "abandon:<reason>" | "advisor"

  // collected form data (mirrors widget.STEP_ACTIONS targets)
  formData: {
    coverage:  { selected: [] as string[] }, // multi
    insured:   null as string | null,        // single
    dob:       "",
    sv:        "",
    svFilter:  "",
    tariff:    null as string | null,
    health:    "no",
    email: "", firstName: "", lastName: "",
    sport: "no", doctor: "", height: "", weight: "",
    consentTos: false, consentPrivacy: false,
  },

  // validation (mirrors Angular ng-invalid/ng-touched)
  errors:  {} as Record<string, string>,
  touched: {} as Record<string, boolean>,

  // S3 SV CDK overlay
  svSelect: { open: false },

  // S4 tariff table
  tooltipsOpen: {} as Record<string, boolean>,
  provisionalPrice: null as number | null,

  // S7 final price
  finalPrice: null as number | null,
};
```

### 2.2 Schema — `sim_loop/replay/src`

The wizard is **5 `Step` elements**, each shown only when `index === currentStepIndex`.
The `NavRow` is part of each step (Weiter/Zurück per-step config). Static tariff/SV
data is in `sim_loop/replay/src*.json` and pulled into the schema at build time
via a tiny `loadSchema()` helper that does the `$ref` inlining (no runtime fetch).

```jsonc
{
  "root": "wizard",
  "elements": {
    "wizard": {
      "type": "Wizard",
      "props": { "current": { "$state": "/currentStepIndex" } },
      "children": ["stepper", "liveRegion", "step1", "step2", "step3", "step4", "step6"]
    },

    "stepper": {
      "type": "ProgressStepper",
      "props": {
        "phases": ["Angaben", "Produkt", "Empfehlung", "Abschluss"],
        "current": { "$state": "/currentStepIndex" }
      }
    },

    "liveRegion": {
      "type": "StepLiveRegion",
      "props": { "heading": { "$state": "/headingForCurrentStep" } }
    },

    /* ───────── S1 — COVERAGE_TYPE (checkbox multi-select) ───────── */
    "step1": {
      "type": "Step",
      "props": { "id": "S1_COVERAGE_TYPE", "index": 0,
                 "current": { "$state": "/currentStepIndex" },
                 "phase": "Angaben",
                 "heading": "Wo möchten Sie abgesichert sein?" },
      "children": ["s1Heading", "s1Instruction", "s1Group", "s1Nav"]
    },
    "s1Heading":     { "type": "Heading",     "props": { "text": "Wo möchten Sie abgesichert sein?", "level": 2 } },
    "s1Instruction": { "type": "Instruction", "props": { "text": "Bitte wählen Sie eine oder mehrere Optionen." } },
    "s1Group": {
      "type": "ChoiceCardGroup",
      "props": {
        "mode": "checkbox",
        "group": "coverage",
        "items": [
          { "id": "bei_arztbesuchen", "title": "Bei Arztbesuchen", "icon": "user-md",
            "features": ["Kassen-, Wahl- oder Privatärzt:in wählen",
                         "Schul- und Alternativmedizin",
                         "Telemedizin – Arztbesuch bequem von Zuhause"] },
          { "id": "im_krankenhaus", "title": "Im Krankenhaus", "icon": "hospital",
            "features": ["Öffentliches Spital oder Privatklinik wählen",
                         "Komfort im Zweibettzimmer",
                         "Operationstermin zeitlich flexibel planen"] }
        ]
      }
    },
    "s1Nav": {
      "type": "NavRow",
      "props": {
        "showBack": false,
        "onBack":  { "action": "cancelWizard", "params": { "reason": "abbrechen_s1" } },
        "onNext":  { "action": "validateStep", "params": { "step": "S1_COVERAGE_TYPE" } },
        "nextDisabled": { "$state": "/derived/s1NextDisabled" }
      }
    },

    /* ───────── S2 — INSURED_PERSONS (radio single-select) ───────── */
    "step2": {
      "type": "Step",
      "props": { "id": "S2_INSURED_PERSONS", "index": 1,
                 "current": { "$state": "/currentStepIndex" },
                 "phase": "Angaben",
                 "heading": "Wer soll versichert werden?" },
      "children": ["s2Heading", "s2Instruction", "s2Group", "s2Nav"]
    },
    "s2Heading":     { "type": "Heading",     "props": { "text": "Wer soll versichert werden?", "level": 2 } },
    "s2Instruction": { "type": "Instruction", "props": { "text": "Bitte wählen Sie eine Option." } },
    "s2Group": {
      "type": "ChoiceCardGroup",
      "props": {
        "mode": "radio", "group": "insured",
        "items": [
          { "id": "ich_selbst", "title": "Ich selbst", "features": [] },
          { "id": "andere_personen", "title": "Andere Personen", "features": [] }
        ]
      }
    },
    "s2Nav": {
      "type": "NavRow",
      "props": {
        "showBack": true,
        "onBack":   { "action": "prevStep", "params": {} },
        "onNext":   { "action": "validateStep", "params": { "step": "S2_INSURED_PERSONS" } },
        "nextDisabled": { "$state": "/derived/s2NextDisabled" }
      }
    },

    /* ───────── S3 — PERSONAL_INFO (DOB + searchable SV) ───────── */
    "step3": {
      "type": "Step",
      "props": { "id": "S3_PERSONAL_INFO", "index": 2,
                 "current": { "$state": "/currentStepIndex" }, "phase": "Angaben",
                 "heading": "Um eine voraussichtliche individuelle Prämie für Sie zu berechnen, benötigen wir:" },
      "children": ["s3Heading", "s3Instruction", "s3Dob", "s3Sv", "s3Nav"]
    },
    "s3Heading":     { "type": "Heading",     "props": { "text": "Um eine voraussichtliche individuelle Prämie für Sie zu berechnen, benötigen wir:", "level": 2 } },
    "s3Instruction": { "type": "Instruction", "props": { "text": "Bitte füllen Sie alle Felder aus." } },
    "s3Dob": {
      "type": "DatePicker",
      "props": {
        "label":   "Geburtsdatum",
        "value":   { "$state": "/formData/dob" },
        "error":   { "$state": "/errors/dob" },
        "touched": { "$state": "/touched/dob" },
        "onChange": { "action": "typeDob",     "params": {} },
        "onFocus":  { "action": "focusField",  "params": { "field": "date_of_birth" } },
        "onBlur":   { "action": "blurField",   "params": { "field": "date_of_birth" } }
      }
    },
    "s3Sv": {
      "type": "SearchableSelect",
      "props": {
        "label":       "Sozialversicherung",
        "placeholder": "Bitte treffen Sie eine Auswahl",
        "value":       { "$state": "/formData/sv" },
        "error":       { "$state": "/errors/sv" },
        "touched":     { "$state": "/touched/sv" },
        "options":     ["ÖGK", "BVAEB-OEB", "SVS Landwirtschaft",
                        "SVS gew.Wirtschaft Sach", "BVAEB-EB", "KFA Wien,NÖ,Sbg,Ktn"],
        "onOpen":      { "action": "openSvSelect",  "params": {} },
        "onPick":      { "action": "pickSv",        "params": {} },
        "onClose":     { "action": "closeSvSelect", "params": {} },
        "onFilter":    { "action": "filterSv",      "params": {} }
      }
    },
    "s3Nav": {
      "type": "NavRow",
      "props": {
        "showBack": true,
        "onBack":   { "action": "prevStep", "params": {} },
        "onNext":   { "action": "validateStep", "params": { "step": "S3_PERSONAL_INFO" } }
      }
    },

    /* ───────── S4 — TARIFF_SELECT (price wall #1) ───────── */
    "step4": {
      "type": "Step",
      "props": { "id": "S4_TARIFF_SELECT", "index": 3,
                 "current": { "$state": "/currentStepIndex" }, "phase": "Produkt",
                 "heading": "Welche Leistungen soll Ihre Privatarzt-Versicherung abdecken?" },
      "children": ["s4Heading", "s4Table", "s4Tooltips", "s4Sticky", "s4Nav"]
    },
    "s4Heading": { "type": "Heading", "props": { "text": "Welche Leistungen soll Ihre Privatarzt-Versicherung abdecken?", "level": 2 } },
    "s4Table": {
      "type": "TariffTable",
      "props": {
        "tariffs":  /* $ref: data/tariffs.json */ [
          { "id": "start",    "name": "Start",     "price": 38.74,  "online": true,  "maxYear": 1400 },
          { "id": "optimal",  "name": "Optimal",   "price": 68.14,  "online": true,  "maxYear": 2800 },
          { "id": "opt_plus", "name": "Opt. Plus", "price": 96.66,  "online": false, "maxYear": 4200 },
          { "id": "premium",  "name": "Premium",   "price": 140.15, "online": false, "maxYear": 8400 }
        ],
        "selected":       { "$state": "/formData/tariff" },
        "onPickTariff":   { "action": "selectTariff", "params": {} },
        "onPremiumClick": { "action": "premiumClick", "params": {} }
      }
    },
    "s4Tooltips": {
      "type": "Stack",
      "props": { "direction": "row", "gap": 8 },
      "children": ["tipHB", "tipArzt", "tipMed", "tipTher", "tipHilf", "tipAugen"]
    },
    "tipHB":   { "type": "CoverageRowTooltip", "props": { "row": "hoechstbetrag",  "body": "Jährlicher Erstattungshöchstbetrag", "open": { "$state": "/tooltipsOpen/hoechstbetrag" },  "onOpen": { "action": "openTooltip", "params": { "row": "hoechstbetrag" } } } },
    "tipArzt": { "type": "CoverageRowTooltip", "props": { "row": "arztleistungen", "body": "Honorare für Fachärzt:innen",          "open": { "$state": "/tooltipsOpen/arztleistungen" }, "onOpen": { "action": "openTooltip", "params": { "row": "arztleistungen" } } } },
    "tipMed":  { "type": "CoverageRowTooltip", "props": { "row": "medikamente",    "body": "Verschreibungspflichtige Medikamente",  "open": { "$state": "/tooltipsOpen/medikamente" },    "onOpen": { "action": "openTooltip", "params": { "row": "medikamente" } } } },
    "tipTher": { "type": "CoverageRowTooltip", "props": { "row": "therapien",      "body": "Physio / Ergo / Logopädie",             "open": { "$state": "/tooltipsOpen/therapien" },      "onOpen": { "action": "openTooltip", "params": { "row": "therapien" } } } },
    "tipHilf": { "type": "CoverageRowTooltip", "props": { "row": "hilfsmittel",    "body": "Brillen, Orthesen, Hörgeräte",          "open": { "$state": "/tooltipsOpen/hilfsmittel" },    "onOpen": { "action": "openTooltip", "params": { "row": "hilfsmittel" } } } },
    "tipAugen":{ "type": "CoverageRowTooltip", "props": { "row": "augen_op",       "body": "Refraktive Augenchirurgie",             "open": { "$state": "/tooltipsOpen/augen_op" },       "onOpen": { "action": "openTooltip", "params": { "row": "augen_op" } } } },
    "s4Sticky": {
      "type": "StickyOfferBar",
      "props": { "label": "Unser Angebot", "price": { "$state": "/provisionalPrice" } }
    },
    "s4Nav": {
      "type": "NavRow",
      "props": {
        "showBack": true,
        "onBack":   { "action": "prevStep", "params": {} },
        "onNext":   { "action": "validateStep", "params": { "step": "S4_TARIFF_SELECT" } },
        "nextDisabled": { "$state": "/derived/s4NextDisabled" }
      }
    },

    /* ───────── S6 — PERSONAL_DATA + HEALTH (+ S7 final price inline) ───────── */
    "step6": {
      "type": "Step",
      "props": { "id": "S6_PERSONAL_DATA", "index": 4,
                 "current": { "$state": "/currentStepIndex" }, "phase": "Abschluss",
                 "heading": "Angaben zu Ihrer Person" },
      "children": ["s6Heading", "s6Health", "s6Final", "s6Nav"]
    },
    "s6Heading": { "type": "Heading", "props": { "text": "Angaben zu Ihrer Person", "level": 2 } },
    "s6Health": {
      "type": "HealthForm",
      "props": {
        "values":   { "$state": "/formData" },
        "errors":   { "$state": "/errors" },
        "onField":  { "action": "setField",      "params": {} },
        "onSubmit": { "action": "submitHealth",  "params": {} }
      }
    },
    "s6Final": {
      "type": "FinalPrice",
      "props": {
        "price":                 { "$state": "/finalPrice" },
        "deltaFromProvisional":  { "$state": "/derived/priceDelta" }
      }
    },
    "s6Nav": {
      "type": "NavRow",
      "props": {
        "showBack": true,
        "onBack":   { "action": "prevStep",      "params": {} },
        "onNext":   { "action": "finishConvert", "params": {} },
        "nextDisabled": { "$state": "/derived/s6NextDisabled" }
      }
    }
  }
}
```

### 2.3 Actions (handlers) and what they emit

Handlers live in `sim_loop/replay/src`. They mutate the funnel store **and**
emit ActivityLog events via the injected `Recorder`. See §3 for the exact event map.

```ts
// registry.tsx (excerpt) — defineRegistry(funnelCatalog, makeImpls({ recorder, transitions }))
actions: {
  toggleCoverage: ({ id }, setState, state) => {
    const prev: string[] = state["/formData/coverage/selected"] ?? [];
    const next = prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id];
    setState("/formData/coverage/selected", next);
    recorder.emit("select", id, /* value */ next.includes(id));
  },
  pickInsured: ({ id }, setState) => {
    setState("/formData/insured", id);
    recorder.emit("select", id);
  },
  typeDob: ({ value }, setState) => {
    setState("/formData/dob", value);
    setState("/touched/dob", true);
    if (value) recorder.emit("keystroke", "date_of_birth", value.length);
  },
  openSvSelect: (_p, setState) => {
    setState("/svSelect/open", true);
    recorder.emit("dropdown_open", "sv_number");
  },
  pickSv: ({ value }, setState) => {
    setState("/formData/sv", value);
    setState("/svSelect/open", false);
    recorder.emit("select", "sv_number", value);
  },
  selectTariff: ({ id }, setState, state) => {
    const tariff = state["/tariffs"].find((t: any) => t.id === id);
    setState("/formData/tariff", id);
    setState("/provisionalPrice", tariff.price);
    recorder.emit("price_reveal", id, tariff.price);
    recorder.emit("select", id);
  },
  premiumClick: ({ id }, setState) => {
    recorder.emit("premium_click", id);
    // does NOT advance — user must pick an online tariff to finish online
  },
  openTooltip: ({ row }, setState, state) => {
    setState(`/tooltipsOpen/${row}`, !(state[`/tooltipsOpen/${row}`] ?? false));
    recorder.emit("tooltip_open", row);
  },
  validateStep: ({ step }, setState, state) => {
    const { errors, ok } = transitions.validate(step, state);
    setState("/errors", errors);
    Object.keys(errors).forEach(field =>
      recorder.emit("validation_error", field, errors[field]));
    if (ok) transitions.advance(step, state, setState, recorder);
  },
  prevStep: (_p, setState, state) => {
    setState("/currentStepIndex", Math.max(0, state["/currentStepIndex"] - 1));
    recorder.emit("nav_back");
  },
  cancelWizard: ({ reason }, setState) => {
    setState("/terminal", `abandon:${reason}`);
    recorder.emit("abandon", null, reason);
  },
  routeAdvisor: ({ reason }, setState) => {
    setState("/terminal", "advisor");
    recorder.emit("abandon", null, `advisor_route(${reason})`);
  },
  finishConvert: (_p, setState) => {
    setState("/terminal", "convert");
    recorder.emit("convert", null, "online_purchase");
  },
  // …focusField, blurField, filterSv, setField, submitHealth, submitClosing analogous
}
```

### 2.4 Transitions / state machine — `sim_loop/replay/src`

A pure module so it's testable in isolation (matches `widget.py.legal_events`).

```ts
// transitions.ts
import { Recorder } from "../capture.js";

const STEP_ORDER = ["S1_COVERAGE_TYPE", "S2_INSURED_PERSONS",
                    "S3_PERSONAL_INFO", "S4_TARIFF_SELECT", "S6_PERSONAL_DATA"];

const DDMMYYYY = /^([0-2]\d|3[01])\.(0\d|1[0-2])\.(19|20)\d\d$/;
const ONLINE_TARIFFS = new Set(["start", "optimal"]);

export const transitions = {
  validate(step, state) {
    const errors: Record<string, string> = {};
    switch (step) {
      case "S1_COVERAGE_TYPE": {
        const sel: string[] = state["/formData/coverage/selected"] ?? [];
        if (sel.length === 0) errors.coverage = "Bitte wählen Sie eine oder mehrere Optionen.";
        return { errors, ok: sel.length > 0 };
      }
      case "S2_INSURED_PERSONS": {
        const v = state["/formData/insured"];
        if (!v) errors.insured = "Bitte wählen Sie eine Option.";
        return { errors, ok: !!v };
      }
      case "S3_PERSONAL_INFO": {
        const dob = state["/formData/dob"] ?? "";
        const sv  = state["/formData/sv"]  ?? "";
        if (!DDMMYYYY.test(dob)) errors.dob = "Bitte geben Sie das Datum im Format TT.MM.JJJJ ein.";
        if (!sv)                 errors.sv  = "Bitte wählen Sie Ihre Sozialversicherung.";
        return { errors, ok: Object.keys(errors).length === 0 };
      }
      case "S4_TARIFF_SELECT": {
        const t = state["/formData/tariff"];
        if (!t) errors.tariff = "Bitte wählen Sie einen Tarif.";
        return { errors, ok: !!t };
      }
      case "S6_PERSONAL_DATA": {
        // closing requires email + consents (health may be "no"); kept minimal here
        if (!state["/formData/email"])         errors.email         = "E-Mail erforderlich.";
        if (!state["/formData/consentTos"])    errors.consentTos    = "Bitte zustimmen.";
        if (!state["/formData/consentPrivacy"])errors.consentPrivacy= "Bitte zustimmen.";
        return { errors, ok: Object.keys(errors).length === 0 };
      }
    }
    return { errors, ok: true };
  },

  /** Side-effecting: drives advance / routing / final-price. */
  advance(step, state, setState, rec: Recorder) {
    switch (step) {
      case "S1_COVERAGE_TYPE": {
        const sel: string[] = state["/formData/coverage/selected"];
        if (sel.includes("im_krankenhaus"))
          return setState("/terminal", "advisor"),
                 rec.emit("abandon", null, "advisor_route(hospital)");
        return this.gotoIndex(1, "S2_INSURED_PERSONS", setState, rec);
      }
      case "S2_INSURED_PERSONS": {
        if (state["/formData/insured"] === "andere_personen")
          return setState("/terminal", "advisor"),
                 rec.emit("abandon", null, "advisor_route(others)");
        return this.gotoIndex(2, "S3_PERSONAL_INFO", setState, rec);
      }
      case "S3_PERSONAL_INFO":
        return this.gotoIndex(3, "S4_TARIFF_SELECT", setState, rec);
      case "S4_TARIFF_SELECT": {
        const t = state["/formData/tariff"];
        if (!ONLINE_TARIFFS.has(t))
          return setState("/terminal", "advisor"),
                 rec.emit("abandon", null, `advisor_route(${t})`);
        return this.gotoIndex(4, "S6_PERSONAL_DATA", setState, rec);
      }
      case "S6_PERSONAL_DATA": {
        // submitHealth computes the FINAL price (= S7) on S6 → fold-in per spec
        const base = state["/provisionalPrice"] ?? 68.14;
        const final = state["/formData/health"] === "yes" ? base + 3.0 : base;
        setState("/finalPrice", final);
        rec.emit("price_reveal", "optimal_final", final);
        // user clicks Abschließen → finishConvert action runs (mapped to "convert")
        return;
      }
    }
  },

  gotoIndex(i, id, setState, rec) {
    setState("/currentStepIndex", i);
    rec.emit("step_enter", null, null);   // step is set automatically by recorder bridge
  },
};
```

---

## 3. 1-1 action-space mapping (the parity gate)

This table is the **contract** between the React twin and `sim_loop/widget.py` /
`contracts.EventType`. CI test (`sim_loop/replay/src`) asserts
the JS action set ⊆ the Python `EventType` vocabulary, and that every event the twin
emits is in `widget.legal_events(step)` for its step.

| Funnel action (UI)              | json-render action                | `EventType` emitted                | Step legality (`legal_events`)            | Status |
|---------------------------------|-----------------------------------|------------------------------------|-------------------------------------------|--------|
| Enter a step (auto)             | (set by `transitions.advance`)    | `step_enter`                       | every step (base)                          | ✓      |
| Toggle coverage checkbox        | `toggleCoverage`                  | `select` (target=coverage id)      | S1 (`select` in per_step)                  | ✓      |
| Pick insured radio              | `pickInsured`                     | `select` (target=insured id)       | S2 (`select` in per_step)                  | ✓      |
| Type DOB                        | `typeDob`                         | `keystroke` (target=date_of_birth) | S3 (`keystroke`)                           | ✓      |
| Focus / blur a field            | `focusField` / `blurField`        | `field_focus` / `field_blur`       | **GAP** — not yet in `legal_events`        | ⚠️     |
| Open SV CDK overlay             | `openSvSelect`                    | `dropdown_open` (target=sv_number) | S3                                         | ✓      |
| Type-filter SV options          | `filterSv`                        | `keystroke` (target=sv_number)     | S3                                         | ✓      |
| Pick SV option                  | `pickSv`                          | `select` (target=sv_number, value) | S3                                         | ✓      |
| Close SV overlay (no pick)      | `closeSvSelect`                   | `cancel_hover` (target=sv_number)  | **GAP** — `cancel_hover` not legal on S3   | ⚠️     |
| Invalid field (Weiter blocked)  | `validateStep`                    | `validation_error` (target, msg)   | S3, S6 (✓); other steps **GAP**            | ⚠️ minor|
| Select tariff (online)          | `selectTariff`                    | `price_reveal` + `select`          | S4                                         | ✓      |
| Click advisory tariff           | `premiumClick`                    | `premium_click`                    | S4                                         | ✓      |
| Hover tariff (dwell ≥ 450ms)    | (Recorder hover; not a JR action) | `price_hover`                      | S4                                         | ✓      |
| Open coverage tooltip           | `openTooltip`                     | `tooltip_open` (target=row)        | S4                                         | ✓      |
| Hover non-tariff element        | (Recorder hover)                  | `hover`                            | base                                       | ✓      |
| S6 submit health → final price  | `submitHealth`                    | `price_reveal` (optimal_final)     | S6 (`price_reveal` added previously)        | ✓      |
| Generic S6 field write          | `setField`                        | `keystroke` (target=field)         | S6                                         | ✓      |
| `Weiter`                        | `validateStep` → `advance`        | `submit` (on success)              | S3, S6 (`submit`); S1/S2 also emit `submit`| ⚠️ minor|
| `Zurück`                        | `prevStep`                        | `nav_back`                         | base                                       | ✓      |
| `Abbrechen` (S1)                | `cancelWizard`                    | `abandon` (reason=abbrechen_s1)    | base                                       | ✓      |
| Close / external link           | (App-level Exit row)              | `abandon` (reason)                 | base                                       | ✓      |
| Tab switch / return             | (Recorder visibilitychange)       | `tab_blur` / `tab_focus`           | base                                       | ✓      |
| No movement ≥ 2.5s              | (Recorder)                        | `idle`                             | base                                       | ✓      |
| Finish purchase                 | `finishConvert`                   | `convert`                          | base (terminal)                            | ✓      |
| Coach widget shown / dismissed  | (overlay layer — see §4)          | `widget_shown` / `widget_dismiss`  | base                                       | ✓      |
| Coach CTA clicked               | (overlay layer)                   | `widget_cta`                       | base                                       | ✓      |

### Parity gaps to fix in `sim_loop/widget.py` (small, no model retrains)

1. Add `EventType.FIELD_FOCUS` and `EventType.FIELD_BLUR` to **base** `legal_events`
   (they're already in `contracts.EventType`; just enlarge the base set). This lets
   the twin honestly report focus/blur on any field without breaking the schema gate.
2. Add `EventType.SUBMIT` to **base** so `Weiter` is legal on every step (today only
   S3/S6 allow it).
3. Either (a) add `EventType.CANCEL_HOVER` to S3, or (b) re-map `closeSvSelect` to
   not emit any event (dropdown closes are not interesting on their own). Pick (b)
   to keep the vocabulary lean.

These three patches are explicitly bot-compatible (we already added `PRICE_REVEAL`
to S6 / `SESSION_GAP` / `TAB_BLUR` / `TAB_FOCUS` to base with the same approach).

---

## 4. Coach overlay layer — on top of the immutable funnel

Per the track rule the funnel **cannot** be modified at runtime. The coach must
therefore *layer on top* — exactly what `EffectorCommand.render: dict` was always
intended to drive. We use a **second `JsonRenderer`** with its own schema (built
per-decision), its own catalog, and its own store. It renders into a portal.

### 4.1 Decision flow

```
backend / sim ──CoachDecision JSON──▶ decisionStream  ──▶ CoachLayer
                                                          │
                                                          ├─ schema = decision.command.render  (json-render spec)
                                                          ├─ catalog = coachCatalog
                                                          ├─ registry = coachRegistry
                                                          └─ store = ephemeral (per-decision)

Effector dispatch (in CoachLayer):
   SHOW_WIDGET      → render overlay schema (CoachCard / Toast / Sheet / Tooltip)
   FOCUS_FIELD      → funnelRefs.get(target)?.focus()
   SCROLL_TO        → funnelRefs.get(target)?.scrollIntoView({ behavior: "smooth" })
   HIGHLIGHT        → funnelRefs.get(target)?.classList.add("coach-highlight")  (auto-removed after timeout)
   AUTOFILL         → effectorBridge.autofill(target, value)   [validates against NEVER_AUTOFILL]
   FILL_SAMPLE      → effectorBridge.fillSample(target, value) [validates against SAMPLE_FILLABLE]
   PRESELECT_TARIFF → effectorBridge.preselect(target)         [online tariffs only]
   SAVE_PROGRESS    → renders a ResumeSheet overlay (consent + email capture)
   NO_ACTION        → emit nothing (most common)
```

`effectorBridge` is the **only** path that mutates the funnel store from outside.
It enforces the same guardrails as `contracts.EffectorCommand.validate()` and
re-emits `widget_shown` so the capture log records the intervention.

### 4.2 Coach catalog — `sim_loop/replay/src/coachWidgets.tsx`

```ts
export const coachCatalog = {
  components: {
    // On-page widgets
    CoachCard:    { props: { intent: "string", headline: "string", body: "string",
                              cta: "string", surface: "string", onCta: "action",
                              onDismiss: "action" } },
    Banner:       { props: { tone: "string", text: "string", onDismiss: "action" } },
    Toast:        { props: { text: "string", duration: "number", onDismiss: "action" } },
    Tooltip:      { props: { anchor: "string", body: "string", onDismiss: "action" } },
    BottomSheet:  { props: { title: "string", onDismiss: "action" } },
    CTA:          { props: { label: "string", onClick: "action", variant: "string" } },
    InfoBlock:    { props: { title: "string", bullets: "string[]" } },
    PriceReframe: { props: { monthly: "number", daily: "number" } },        // "€68/mo = €2.27/day"
    MarketCompare:{ props: { ours: "number", market: "number", note: "string" } },
    CallbackCTA:  { props: { channel: "string", phone: "string", onAccept: "action" } },

    // Off-page surfaces (rendered as a "sent" confirmation in the on-page layer,
    // real delivery is mocked by effectorBridge.deliver(surface, payload))
    EmailResume:  { props: { subject: "string", to: "string", body: "string" } },
    WhatsAppCTA:  { props: { number: "string", message: "string" } },
    Survey:       { props: { question: "string", options: "string[]", onPick: "action" } },
  },
  actions: {
    cta:     { params: { intent: "string" } },                  // user accepted
    dismiss: { params: { intent: "string" } },                  // user closed
    surfacePick: { params: { surface: "string", option: "string" } },
  },
} as const;
```

### 4.3 Registry — `sim_loop/replay/src/coachWidgets.tsx`

Each component is a small focused React file. They all:
1. Receive their props from the `CoachDecision.command.render` schema.
2. Render via a **`createPortal` into `#coach-layer`** (a fixed-position div in
   `index.html`, `z-index: 9000`, `pointer-events: none` on the wrapper and
   `auto` on the actual card so the funnel keeps receiving clicks elsewhere).
3. Call `recorder.emit("widget_shown", intent, { surface })` on mount.
4. Call `recorder.emit("widget_cta", intent)` or `widget_dismiss` on interaction.

Action handlers:

```ts
actions: {
  cta:        ({ intent }, _setState) => recorder.emit("widget_cta",     intent),
  dismiss:    ({ intent }, _setState) => recorder.emit("widget_dismiss", intent),
  surfacePick:({ surface, option }, _setState) =>
                recorder.emit("widget_cta", surface, option),
}
```

### 4.4 Effector bridge — `sim_loop/replay/src/coachWidgets.tsx`

The only place that can write the funnel store from outside the funnel itself.
Mirrors `contracts.EffectorCommand.validate()` 1-1.

```ts
const NEVER_AUTOFILL  = new Set(["sv_number","first_name","last_name","email",
                                  "health_answers","date_of_birth","consent"]);
const SAMPLE_FILLABLE = new Set(["coverage","insured","tariff"]);
const ONLINE_TARIFFS  = new Set(["start","optimal"]);

export function makeEffectorBridge(funnelStore, recorder) {
  return {
    autofill(field, value) {
      if (NEVER_AUTOFILL.has(field))
        throw new Error(`AUTOFILL forbidden on protected field '${field}'`);
      funnelStore.set(`/formData/${field}`, value);
      recorder.emit("widget_shown", "autofill", { field, value });
    },
    fillSample(field, value) {
      if (!SAMPLE_FILLABLE.has(field))
        throw new Error(`FILL_SAMPLE only allowed on ${[...SAMPLE_FILLABLE]}`);
      funnelStore.set(`/formData/${field}`, value);
      recorder.emit("widget_shown", "fill_sample", { field, value });
    },
    preselect(tariffId) {
      if (!ONLINE_TARIFFS.has(tariffId))
        throw new Error(`PRESELECT_TARIFF: online tariffs only`);
      funnelStore.set("/formData/tariff", tariffId);
      recorder.emit("widget_shown", "preselect_tariff", { tariff: tariffId });
    },
    highlight(elementId, ms = 1500) {
      const el = funnelRefs.get(elementId);
      if (!el) return;
      el.classList.add("coach-highlight");
      setTimeout(() => el.classList.remove("coach-highlight"), ms);
      recorder.emit("widget_shown", "highlight", { target: elementId });
    },
    focus(elementId)  { funnelRefs.get(elementId)?.focus(); },
    scrollTo(id)      { funnelRefs.get(id)?.scrollIntoView({ behavior: "smooth" }); },
    deliver(surface, payload) {
      // mocked off-page delivery (email/whatsapp/survey/landing/advisor_booking).
      // Real impl: POST to a local mock endpoint that prints to console + writes
      // to _local/captures/sent/<surface>-<ts>.json. The on-page layer still
      // renders a "sent" confirmation so the user sees it happened.
      recorder.emit("widget_shown", surface, payload);
    },
  };
}
```

### 4.5 CoachLayer — `sim_loop/replay/src/coachWidgets.tsx`

```tsx
export function CoachLayer({ stream, funnelStore, recorder }) {
  const [active, setActive] = useState<CoachDecision[]>([]);
  const bridge   = useMemo(() => makeEffectorBridge(funnelStore, recorder), []);
  const registry = useMemo(() => coachRegistry({ recorder, bridge }), []);

  useEffect(() => stream.subscribe(onDecision), [stream]);

  function onDecision(d: CoachDecision) {
    const eff = d.command.effector;
    // 1) side-effecting effectors execute immediately
    if (eff === "focus_field")      bridge.focus(d.command.target);
    if (eff === "scroll_to")        bridge.scrollTo(d.command.target);
    if (eff === "highlight")        bridge.highlight(d.command.target);
    if (eff === "autofill")         bridge.autofill(d.command.target, d.command.payload?.value);
    if (eff === "fill_sample")      bridge.fillSample(d.command.target, d.command.payload?.value);
    if (eff === "preselect_tariff") bridge.preselect(d.command.target);
    if (eff === "save_progress")    bridge.deliver("save_progress", d.command.payload);
    // 2) renderable overlays (show_widget OR any off-page surface) get added
    if (eff === "show_widget" || isOffPageSurface(d)) setActive(a => [...a, d]);
  }

  return createPortal(
    <div className="coach-layer">
      {active.map(d => (
        <JsonRenderer
          key={d.command.target ?? d.reasoning.slice(0, 24)}
          schema={d.command.render}              // CoachDecision.command.render IS a json-render spec
          registry={registry}
          initialState={{ intent: d.command.payload?.intent,
                           surface: d.command.payload?.surface ?? "on_page" }}
          onStateChange={() => recorder.emit("widget_shown", d.command.payload?.intent)}
        />
      ))}
    </div>,
    document.getElementById("coach-layer")!
  );
}
```

### 4.6 Surface types (1-1 with `docs/PIPELINE_PLAN.md` §4)

| Surface              | How it renders in the React twin                                   | Outcome emitted                            |
|----------------------|--------------------------------------------------------------------|--------------------------------------------|
| `on_page` widget     | CoachCard / Toast / Tooltip via `CoachLayer` portal                | `widget_cta` / `widget_dismiss`            |
| `email`              | `EmailResume` "sent" confirmation card; `bridge.deliver("email")`  | `widget_shown` (surface=email); later `widget_cta` if user clicks resume link in mock inbox |
| `whatsapp`           | `WhatsAppCTA` overlay with QR + tel link; `bridge.deliver("wa")`   | `widget_cta` on accept                     |
| `landing_page`       | Renders an embedded landing page in a modal sheet                  | `widget_cta` on funnel re-entry            |
| `feedback_form`      | `Survey` overlay (3 options)                                        | `widget_cta` (with option), `widget_dismiss`|
| `advisor_booking`    | Existing UNIQA advisor-booking iframe-style placeholder            | `abandon:advisor_route(booking)` (terminal)|

### 4.7 Decision stream — `sim_loop/replay/src/coachWidgets.tsx`

Pluggable source so demo, local dev, and a future Python backend all fit:

```ts
export interface DecisionStream {
  subscribe(cb: (d: CoachDecision) => void): () => void;
}

export const localMockStream: DecisionStream = /* arrays of canned decisions */;
export const wsStream = (url: string): DecisionStream => /* websocket */;
export const httpPollStream = (url: string, ms = 1000): DecisionStream => /* poll */;
```

This keeps the React app decoupled from the Python coach: today, `localMockStream`
plays a few hand-authored decisions (great for showing each effector); tomorrow,
`wsStream("ws://localhost:8765")` connects to a small bridge that wraps
`coach/coach_io.RuleCoachModel.decide()`.

### 4.8 Why this satisfies the track rule

- The funnel schema (`schema/funnel.json`) is **never patched** by the coach.
- All coach UI is **portal-overlaid** on top — independent React tree, independent
  store, independent registry.
- The only writes the coach can perform on the funnel data go through the
  `effectorBridge`, which is the runtime analog of `EffectorCommand.validate()` —
  identical guardrails (`NEVER_AUTOFILL`, `SAMPLE_FILLABLE`, online-tariffs-only)
  enforced at runtime, not in the schema.

---

## 5. File plan (`sim_loop/replay/src/`)

> Keep files small and topical. No file should exceed ~200 lines. Each component
> in `twin/components/` and `coach/components/` is one file; pure & testable.

```
sim_loop/replay/src/
├── main.jsx                       (unchanged — mounts <App/>)
├── App.jsx                        REWRITE: mounts <FunnelTwin/> + <CoachLayer/>;
│                                   sidebar persona picker + capture log;
│                                   exit-row (close / external link) preserved.
├── styles.css                     KEEP, extend with Reef-parity tokens + `.coach-layer`
├── capture.js                     KEEP (Recorder already correct)
│
├── twin/                          ── IMMUTABLE FUNNEL ──
│   ├── FunnelTwin.tsx             top-level: <JsonRenderer schema registry store/>
│   ├── catalog.ts                 (§1.1 — type-only catalog)
│   ├── registry.tsx               defineRegistry(funnelCatalog, makeImpls({recorder, transitions, funnelRefs}))
│   ├── store.ts                   useStateStore + initialState (§2.1)
│   ├── captureBridge.ts           wires Recorder ↔ store.onStateChange  (for field_focus etc.)
│   ├── transitions.ts             pure state machine + per-step validators (§2.4)
│   ├── funnelRefs.ts              Map<elementId, HTMLElement> registered by components;
│   │                                used by coach `focus/scrollTo/highlight`
│   ├── schema/
│   │   └── funnel.json            (§2.2 — the full wizard schema)
│   ├── data/
│   │   ├── tariffs.json           4 tariffs (mirrors widget.py TARIFFS)
│   │   ├── sv_options.json        6 options (mirrors widget.py SV_OPTIONS)
│   │   ├── coverage_rows.json     6 tooltip rows + body copy
│   │   ├── i18n.de.json           all German strings ↔ used by Heading/Instruction
│   │   └── errors.de.json         verbatim Reef error messages
│   ├── components/                one file each, small + readable
│   │   ├── ProgressStepper.tsx
│   │   ├── StepLiveRegion.tsx
│   │   ├── Step.tsx
│   │   ├── Wizard.tsx
│   │   ├── ChoiceCardGroup.tsx
│   │   ├── ChoiceCard.tsx
│   │   ├── DatePicker.tsx
│   │   ├── SearchableSelect.tsx   (CDK-overlay-style portal listbox)
│   │   ├── TextField.tsx
│   │   ├── HelperTextError.tsx
│   │   ├── TariffTable.tsx
│   │   ├── TariffColumn.tsx
│   │   ├── CoverageRowTooltip.tsx
│   │   ├── HealthForm.tsx
│   │   ├── FinalPrice.tsx
│   │   ├── ClosingForm.tsx
│   │   ├── Button.tsx
│   │   ├── NavRow.tsx
│   │   ├── StickyOfferBar.tsx
│   │   └── Stack.tsx / Heading.tsx / Instruction.tsx
│   └── __tests__/
│       ├── parity.test.ts         (§3 table: every emitted EventType ∈ legal_events(step))
│       ├── transitions.test.ts    (gates, routing, validation messages verbatim)
│       └── schema.test.ts         (every element id referenced exists; types match catalog)
│
├── coach/                         ── MUTABLE OVERLAY ──
│   ├── CoachLayer.tsx             (§4.5)
│   ├── catalog.ts                 (§4.2)
│   ├── registry.tsx               (§4.3)
│   ├── effectorBridge.ts          (§4.4 — runtime analog of EffectorCommand.validate)
│   ├── decisionStream.ts          (§4.7 — local mock / ws / http)
│   ├── components/
│   │   ├── CoachCard.tsx
│   │   ├── Banner.tsx
│   │   ├── Toast.tsx
│   │   ├── Tooltip.tsx
│   │   ├── BottomSheet.tsx
│   │   ├── CTA.tsx
│   │   ├── InfoBlock.tsx
│   │   ├── PriceReframe.tsx
│   │   ├── MarketCompare.tsx
│   │   ├── CallbackCTA.tsx
│   │   ├── EmailResume.tsx
│   │   ├── WhatsAppCTA.tsx
│   │   └── Survey.tsx
│   ├── decisions/                  hand-authored CoachDecisions for the demo
│   │   ├── s4_price_reframe.json
│   │   ├── s4_premium_clicked.json
│   │   ├── s4_peter_callback.json
│   │   ├── s6_form_helper.json
│   │   └── s7_save_progress.json
│   └── __tests__/
│       ├── effector_guards.test.ts (NEVER_AUTOFILL, SAMPLE_FILLABLE, online-only)
│       └── overlay_render.test.tsx (each decision schema renders + emits widget_shown)
│
├── index.html                     +<div id="coach-layer"></div>
└── README.md                      run instructions + how to plug a real coach stream
```

### File-by-file responsibilities

- `FunnelTwin.tsx` (~80 lines): creates the `funnelStore`, builds the `funnelRegistry`
  with the injected `Recorder`, mounts a single `<JsonRenderer>` with `schema/funnel.json`.
- `registry.tsx` (~150 lines): `defineRegistry(funnelCatalog, { components: { … }, actions: { … } })`.
  Each component impl is a one-line import from `components/<Name>.tsx`.
- Each `components/*.tsx` file is one component, props-typed from the catalog, with
  the Reef visual treatment in CSS. No business logic, no event emission — those
  live entirely in the action handlers.
- `transitions.ts` is **framework-free** (no React import) so it can be unit-tested
  with the same fixtures the Python `transitions` tests use (port `test_scope.py`
  routing cases verbatim).
- `captureBridge.ts` wires `store.onStateChange` to derive `field_focus`/`field_blur`
  events from `/touched/*` flips when needed (most events are already emitted by
  action handlers; this catches the rest).

---

## 6. Open risks / decisions for the next step

1. **Field focus/blur in `legal_events`** — recommend adding `FIELD_FOCUS`/`FIELD_BLUR`
   to the base set (one-liner in `widget.py`); see §3 parity gaps. Without it, focusing
   a field would currently fail the schema gate.
2. **SubmitOn S1/S2** — same minor fix: add `SUBMIT` to base (currently only S3/S6).
3. **Real-time `$state` performance** — every `setState` re-renders elements bound
   to the changed path. Two mitigations: (a) use granular paths (`/formData/dob` not
   `/formData`), (b) keep `derived/*` flags pre-computed in `captureBridge` rather
   than recomputed in render. Documented; flagged for implementer.
4. **`closeSvSelect` event** — recommend dropping the event entirely (no `cancel_hover`
   needed on S3); the dropdown close is captured implicitly by the subsequent action.
5. **Tariff price formatting** — twin uses `€38.74` (point) to match `widget.py`;
   the real funnel renders `€38,74` (comma) for DE locale. The implementer can use
   `Intl.NumberFormat("de-AT", { style: "currency", currency: "EUR" })` at the
   component layer — the value in state stays a number so action emission is unchanged.

---

**Summary**

1. Twin = `@json-render/react` runtime: typed `funnelCatalog`, React `funnelRegistry`,
   pure `transitions.ts`, JSON `schema/funnel.json` with `$state` bindings — one
   `JsonRenderer`, no schema patching at runtime.
2. Every Reef component captured in `UNIQA_FUNNEL_SPEC.md` has an explicit catalog
   entry and a small dedicated React file under `twin/components/`.
3. The action-space mapping table (§3) is the parity contract; only 3 small fixes
   needed in `sim_loop/widget.py` (`FIELD_FOCUS`/`FIELD_BLUR`/`SUBMIT` into base
   `legal_events`) — no Python model retrains.
4. Coach is a **separate `JsonRenderer`** with its own catalog + portal layer
   driven by streamed `CoachDecision` JSON; `EffectorCommand.render` *is* the
   overlay schema.
5. `effectorBridge` is the runtime analog of `EffectorCommand.validate()` —
   only place that can write the funnel store from outside, enforcing
   `NEVER_AUTOFILL` / `SAMPLE_FILLABLE` / online-tariffs guardrails.
6. Concrete file plan (`sim_loop/replay/src` + `sim_loop/replay/src/coachWidgets.tsx`) keeps each
   file under ~200 lines, with tests for parity, transitions, and effector
   guards before any UI work lands.
