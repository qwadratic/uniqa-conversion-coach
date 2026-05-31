"""Coach model — observes the shared event feed and decides ONE json-render widget.

Two modes:
  - "skip"   : always NO_ACTION, no LLM call (the control arm)
  - "active" : LLM policy; emits {persona_belief, ..., command} where `command` is a
               typed JSON-RENDER widget spec (effector id + fe_pattern + title/body/cta).

NEW (event-feed refactor):
  observe(feed, step) — called AFTER EVERY persona micro-turn.
    Takes the raw shared feed (source=user|coach events), filters internally to
    coach-visible events (drops thought/state/feeling; keeps observable user events +
    prior coach events), and returns NO_ACTION or ONE effector appended to the feed.
    The coach does NOT "take the turn"; injecting an event != the turn passing to it.

  decide(filtered_log, step) still works as the lower-level API.

Information isolation: the coach NEVER sees persona thoughts, mental-state vars, feeling,
persona label, or S6 health data — only the observable event log.

The COMPLETE effector library (ported from coach/interventions.py CATALOG) is the single
source of truth for both the prompt menu and the front-end json-render registry. Each effector
carries a `category` and a default `fe_pattern` (the overlay shape) so the renderer can draw it.
"""
from __future__ import annotations
import json
from llm import chat, extract_json

# ── the FULL effector library: id, category, default fe_pattern, when-apt hint, persona-facing body ──
# fe_pattern ∈ FE_PATTERNS; the coach may override it, else this default is used by the renderer.
EFFECTOR_LIBRARY: list[dict] = [
    # price
    {"id": "price_reframe", "category": "price", "fe": "price_chip", "apt": "S4,S7", "who": "",
     "body": "€{m}/mo ≈ €{d}/day — less than a coffee, for what it covers."},
    {"id": "pricing_explain", "category": "price", "fe": "card", "apt": "S4,S7", "who": "",
     "body": "How your premium is built: your age + the tariff. Health answers may adjust the final price — nothing hidden."},
    {"id": "price_preview", "category": "price", "fe": "popover", "apt": "S6", "who": "",
     "body": "Before you fill this: here's the small impact this answer has on your price — no surprises later."},
    {"id": "value_justification", "category": "price", "fe": "card", "apt": "S7", "who": "franz,judith",
     "body": "Your final price is a little higher because it now covers X. Prefer leaner? Here's a comparable cheaper tariff — still fully online."},
    # inform
    {"id": "health_explain", "category": "inform", "fe": "banner", "apt": "S7", "who": "",
     "body": "Your final price moved after the health questions — here's why, and you can still finish fully online."},
    {"id": "upgrade_explain", "category": "inform", "fe": "banner", "apt": "S4", "who": "franz,judith",
     "body": "Premium and Opt.Plus need an advisor — but Optimal is fully completable online right now."},
    {"id": "package_nuance", "category": "inform", "fe": "inline_expand", "apt": "S4", "who": "",
     "body": "Start vs Optimal — the 3 differences that actually matter (limits, refund %, what's included). No upsell."},
    {"id": "coverage_explain", "category": "inform", "fe": "popover", "apt": "S1,S4", "who": "",
     "body": "What the limits mean in real life: e.g. ≈ X specialist visits or a physio series a year."},
    {"id": "coverage_checker", "category": "inform", "fe": "popover", "apt": "S1,S4", "who": "",
     "body": "Is YOUR treatment / doctor covered? Ask here — the one question the form gives you no way to ask."},
    {"id": "faq_cards", "category": "inform", "fe": "card", "apt": "S1,S4,S5", "who": "",
     "body": "Quick answers to what you're probably wondering right now — no need to leave and search."},
    {"id": "feature_highlight", "category": "inform", "fe": "toast", "apt": "S4", "who": "",
     "body": "Recently improved: a concrete upgrade relevant to you (e.g. laser-eye-surgery limit doubled in 2025)."},
    # reassure
    {"id": "trust_signal", "category": "reassure", "fe": "banner", "apt": "any", "who": "",
     "body": "UNIQA — insuring Austria since 1811, AAA-rated."},
    {"id": "social_proof", "category": "reassure", "fe": "banner", "apt": "S4,S7", "who": "judith,peter",
     "body": "Most people with needs like yours chose Optimal this month."},
    {"id": "addon_skip_ok", "category": "reassure", "fe": "banner", "apt": "S5", "who": "",
     "body": "Add-ons are optional — skipping keeps your price exactly as shown, and you can add any later."},
    # engage
    {"id": "quick_quiz", "category": "engage", "fe": "bottom_sheet", "apt": "S1,S4", "who": "peter,judith",
     "body": "Not sure which tariff? Answer 3 quick questions and we'll recommend the right one (≈60s)."},
    {"id": "form_simplify", "category": "engage", "fe": "bottom_sheet", "apt": "S3,S6", "who": "peter",
     "body": "Let's make this short: only the required fields, split into small steps."},
    {"id": "field_defer", "category": "engage", "fe": "popover", "apt": "S3,S6", "who": "",
     "body": "Don't have it handy? Skip this field now and add it later — we'll flag it if it affects your price."},
    {"id": "bucket_input", "category": "engage", "fe": "popover", "apt": "S6", "who": "",
     "body": "Pick a range instead of an exact number — we'll show the price impact per range."},
    {"id": "form_helper", "category": "engage", "fe": "popover", "apt": "S3,S6", "who": "",
     "body": "Your insurance number is on the top-right of your e-card."},
    {"id": "form_explainer", "category": "engage", "fe": "progress_ribbon", "apt": "S3,S6", "who": "",
     "body": "Just ~4 quick fields (~1 min) — we need them to compute your binding price. Then you're done."},
    {"id": "jump_to_pricing", "category": "convert_aid", "fe": "sticky_bar", "apt": "S2,S3", "who": "franz",
     "body": "You're moving fast — skip ahead to your price now? You can fill the optional detail after."},
    {"id": "id_austria_login", "category": "engage", "fe": "card", "apt": "S3,S6", "who": "",
     "body": "Auto-fill your details with ID Austria — no typing your SV number."},
    # convert_aid
    {"id": "preselect_optimal", "category": "convert_aid", "fe": "chip", "apt": "S4", "who": "franz",
     "body": "Optimal is pre-selected as a sensible default — change it anytime."},
    {"id": "upgrade_path", "category": "convert_aid", "fe": "banner", "apt": "S4", "who": "judith",
     "body": "Start now on a lower tariff and upgrade within 3 years — no new health check."},
    # capture
    {"id": "save_progress", "category": "capture", "fe": "sticky_bar", "apt": "S4,S7", "who": "",
     "body": "Not finishing today? Save your progress to an email link and pick up exactly here later."},
    {"id": "email_capture", "category": "capture", "fe": "card", "apt": "S4,S7", "who": "",
     "body": "Want this quote in writing? We'll email you the summary + a short explainer."},
    {"id": "phone_capture", "category": "capture", "fe": "bottom_sheet", "apt": "S4,S7", "who": "",
     "body": "On your phone? Get your quote by text and a callback — no long form on a small screen."},
    # handoff
    {"id": "callback_offer", "category": "handoff", "fe": "bottom_sheet", "apt": "S1,S3", "who": "peter",
     "body": "Prefer a person? Book a free callback — an advisor walks you through it."},
    {"id": "whatsapp_bot", "category": "handoff", "fe": "bottom_sheet", "apt": "S1,S3,S4", "who": "peter",
     "body": "Ask on WhatsApp — we'll answer your questions and send the quote."},
    {"id": "contact_handoff", "category": "handoff", "fe": "bottom_sheet", "apt": "S3,S4,S6", "who": "peter",
     "body": "Don't fill any of this — just leave your email or phone and we'll take it from here."},
    {"id": "voice_questions", "category": "handoff", "fe": "bottom_sheet", "apt": "S1,S3,S4", "who": "peter",
     "body": "Leave a number for a callback, or type your questions and we'll get them answered."},
    {"id": "advisor_handoff", "category": "handoff", "fe": "bottom_sheet", "apt": "S7", "who": "judith",
     "body": "Want a person to confirm before you commit? Talk to an advisor to finish — fully optional."},
]
EFFECTORS = [e["id"] for e in EFFECTOR_LIBRARY]
_BY_ID = {e["id"]: e for e in EFFECTOR_LIBRARY}
FE_PATTERNS = ["price_chip", "card", "popover", "banner", "inline_expand", "toast",
               "bottom_sheet", "progress_ribbon", "sticky_bar", "chip"]
SURFACES = ["on_page", "email", "whatsapp"]


def _menu() -> str:
    return "\n".join(
        f"  - {e['id']} [{e['category']}/{e['fe']}]"
        f"{' apt:' + e['apt'] if e['apt'] != 'any' else ''}"
        f"{' fits:' + e['who'] if e['who'] else ''}: {e['body']}"
        for e in EFFECTOR_LIBRARY)


COACH_SYSTEM = ("""You are the UNIQA Conversion Coach — a DETECTION + DECISION layer on TOP of an
unchangeable insurance calculator. You never edit the funnel; you watch the user's OBSERVABLE
event trace (events: type, target, value, timing, step, + their continue/leave decision) and
decide whether to show ONE helpful widget, and which. You do NOT see who they are, their thoughts,
their mental state, their feeling, or their health (S6) answers. Be the opposite of a nag: most of
the time you WAIT. Reason from the principles below — these are priors, not rigid rules.

## Your goal depends on WHO the user is (persona-dependent conversion target)
- **Judith** (Rising Hybrid): online purchase OR a smooth advisor handoff — BOTH count (advisor-affine; don't force online).
- **Franz** (Online Affine): online purchase ONLY — pushing him to an advisor/offline is a FAILURE. Save budget for the FINAL price.
- **Peter** (Service Affine): a qualified SERVICE CONTACT (callback / WhatsApp / phone) — online purchase is NOT his target; finishing by phone is a WIN.
Detect the persona first, then optimize THEIR right outcome.

## Be SIGNAL-FIRST and CREATIVE (read this)
Lead with the CONCRETE behavioural signals in the activity log — NOT a persona guess. Hesitation is
visible: dwell/hover on the price without selecting, price_hover→cancel_hover (price recoil),
nav_back (second-guessing), scroll_up / re-reads (confusion), idle/pause (stalling), tab_blur /
external_nav (left to compare), exit_intent (about to leave — last chance), validation_error /
field_clear / text_select / copy (a field or term is fighting them). These are your TRIGGERS;
the persona is only a prior. If the log shows no such signal and the user is flowing, WAIT.
The effector menu is a PALETTE, not a rigid persona→screen map: pick whatever genuinely fits the
SIGNAL + situation (even off the usual step/persona), and **write fresh, specific copy** (title/body/
cta) that speaks to exactly what you just saw them do — e.g. name the term they copied, the price
they recoiled from, the field they cleared. A precise, well-timed reaction to a real signal beats a
generic persona-templated card every time.

## Run this every turn (observe → decide)
1. PERSONA BELIEF — confidence over {judith, franz, peter}: FAST mechanical S1→S3 → Franz (don't
   interrupt early; he drops at the FINAL price). SLOW + overwhelm + back-nav + re-edits → Peter.
   Deliberate hover/tooltip/compare → Judith. Sharpen over time; commit ~by S3/S4. Read fast/slow
   RELATIVE to a typical user; act on anomalies.
2. PAINS & FRUSTRATION — price shock (dwell after a price reveal, cancel-hover, exit-intent → wants
   OUT) · form overwhelm (hesitation/back-nav on a long form) · term confusion (text-select/copy,
   tooltip, scroll-up) · compare-intent (tab-away then return) · forgot-a-field. Track frustration 0..1.
3. DROPOUT LIKELIHOOD + INTERVENTION TEMPERATURE — estimate P(bounce this step). Willingness to act
   = temperature rising with dropout × belief-confidence. Low → WAIT. High + a clearly matching
   widget → act. Exit-intent spikes it (last chance). Respect the strict annoyance budget.
4. WIDGET MATCH — pick ONE effector id from the MENU whose purpose addresses the detected pain AND
   serves THIS persona's conversion target, on THIS step. Prefer the least-intrusive fe_pattern that
   works; escalate intrusiveness only with temperature. Choose a surface (mobile/Peter → whatsapp or
   email handoff; else on_page). Write the user-facing `title` + `body` in UNIQA voice (Sie-form ok),
   persona-tailored, and a short `cta` if the widget invites a click.
5. FEEDBACK — prior widget ignored/dismissed → back off, don't repeat. Engaged → you read them right;
   you may follow up once.

## Hypotheses (priors to TEST): H1 fast→Franz (save budget for final price) · H2 slow/overwhelm→Peter
(early handoff: callback_offer/whatsapp_bot/contact_handoff) · H3 research→Judith · H6 confident tariff
pick→high P(convert), don't over-intervene · H8 price-shock→price_reframe/pricing_explain not a nudge ·
H10 jargon copy/select→coverage_explain/coverage_checker on that term · H12 big-form→form_explainer
pre-emptively · H14 price_preview so price is never a surprise · H16 Peter→handoff before the price wall ·
H20 at S5→addon_skip_ok. Each is falsifiable; the trace confirms/refutes it.

THE EFFECTOR MENU (pick ONE `effector` id, or NO_ACTION):
""" + _menu() + """

Available fe_patterns: """ + ", ".join(FE_PATTERNS) + """
Available surfaces: """ + ", ".join(SURFACES) + """

## LEARNED CONSTRAINTS (from the autoresearch loop — HARD, do not override)
- NEVER use `form_simplify` on S4_TARIFF_SELECT or S5_ADDON_SELECT — those are price/comparison
  screens, not forms. `form_simplify` is ONLY for the long form steps S3 and S6.
- Do NOT spend `jump_to_pricing` (or any early nudge) on S3 unless the user is clearly fast/Franz;
  for everyone else, hold the budget for the price walls (S4 first price, S7 final price).

## Output STRICT JSON only — ONE object, no prose outside it:
{
  "persona_belief": {"judith": <0..1>, "franz": <0..1>, "peter": <0..1>},
  "detected_pains": ["<short tags>"],
  "frustration": <0..1>,
  "dropout_likelihood": <0..1>,
  "intervention_temperature": <0..1>,
  "reasoning": "<short chain: signal → persona → why this widget now>",
  "command": {
     "effector": "<one id from the menu, or NO_ACTION>",
     "category": "<the effector's category, or null>",
     "fe_pattern": "<one fe_pattern>",
     "surface": "<on_page|email|whatsapp>",
     "title": "<short headline the user sees>",
     "message": "<the body text the user sees>",
     "cta": "<button label, or empty>",
     "target": "<element/term/tariff or null>"
  },
  "hypotheses": ["<falsifiable expectation about the user's next move>"],
  "value_estimate": <0..1 expected conversion help>
}
WAIT (NO_ACTION) is the default and most common decision. Never exceed the budget. One widget at a
time. If you choose NO_ACTION, set fe_pattern "banner", surface "on_page", title/message/cta empty,
target null.""")


# ── persona prediction → drives the coaching TYPE ────────────────────────────────
_FRICTION = {"nav_back", "validation_error", "field_clear", "rage_click", "idle", "pause"}
_DELIBERATION = {"hover", "tooltip_open", "tab_blur", "external_nav", "slow_mouse", "scroll_up",
                 "price_hover", "cancel_hover", "compare_return", "text_select", "copy"}
_FRANZ_EARLY = {"S3_PERSONAL_INFO", "S4_TARIFF_SELECT", "S5_ADDON_SELECT"}
_HANDOFF = {"advisor_handoff", "callback_offer", "whatsapp_bot", "contact_handoff", "voice_questions"}


def predict_persona(filtered_log: list) -> tuple[str, float, dict]:
    """Cheap persona prediction from the OBSERVABLE log (no labels): fast/clean → franz,
    friction (re-edits/back-nav/idle) → peter, deliberation (hover/tooltip/compare) → judith.
    This drives the coaching TYPE — Franz-silence vs proactive Peter-handoff."""
    if not filtered_log:
        return "unknown", 0.0, {}
    types = [str(e.get("type", "")) for e in filtered_log]
    n = len(types); steps = len(set(e.get("step") for e in filtered_log)) or 1
    fr = sum(t in _FRICTION for t in types)
    de = sum(t in _DELIBERATION for t in types)
    dens = n / steps                                   # events per step (Franz = lean/mechanical)
    peter = min(1.0, fr / 3.0)
    judith = min(1.0, de / 3.0)
    franz = max(0.0, 1.0 - 0.4 * fr - 0.3 * de) * (1.0 if dens <= 4 else 0.6)
    scores = {"judith": round(judith, 2), "franz": round(franz, 2), "peter": round(peter, 2)}
    label = max(scores, key=scores.get)
    return label, scores[label], scores


class CoachModel:
    def __init__(self, mode: str = "skip", model: str | None = None,
                 budget: int = 2, temperature: float = 0.4, system: str | None = None):
        assert mode in ("skip", "active")
        self.mode = mode
        self.model = model
        self.budget = budget
        self.temperature = temperature
        self.system = system or COACH_SYSTEM   # policy override (autoresearch varies this)
        self.used = 0

    # S1–S2 are pure DETECTION steps (no churn) — observe only, preserve the budget for the
    # steps where help matters: the forms (S3/S6) and the price walls (S4 + the final price).
    DETECTION_ONLY = {"S1_COVERAGE_TYPE", "S2_INSURED_PERSONS"}

    def decide(self, filtered_log: list, step: str, form_state: dict | None = None) -> dict:
        if self.mode == "skip" or self.used >= self.budget:
            return self._noop("control-skip" if self.mode == "skip" else "budget-exhausted")
        if step in self.DETECTION_ONLY:
            return self._noop("detection-only-step")
        # 1) PREDICT the persona from the observable log — used both as a prompt prior and a gate
        plabel, pconf, pscores = predict_persona(filtered_log)
        obs = {"current_step": step, "annoyance_budget_left": self.budget - self.used,
               "surfaces_available": SURFACES, "activity_log": filtered_log,
               "form_state": form_state or {},
               "predicted_persona": {
                   "label": plabel, "confidence": pconf, "scores": pscores,
                   "policy": "franz → stay SILENT through S3-S5 and NEVER hand off to a human — save the "
                             "budget for the final price (he only drops there); peter → be PROACTIVE with a "
                             "callback / WhatsApp / contact handoff EARLY (a service contact IS his win); "
                             "judith → reassure + offer a graceful advisor option."}}
        user = "OBSERVATION (activity log only — decide your move):\n" + json.dumps(obs, ensure_ascii=False)
        try:
            raw = chat([{"role": "system", "content": self.system},
                        {"role": "user", "content": user}],
                       model=self.model, temperature=self.temperature, max_tokens=600)
            d = extract_json(raw)
        except Exception as e:
            return self._noop(f"coach-error:{type(e).__name__}")
        cmd = d.get("command", {}) or {}
        eff = cmd.get("effector", "NO_ACTION")
        if eff not in _BY_ID:
            eff = "NO_ACTION"
        meta = _BY_ID.get(eff, {})
        fe = cmd.get("fe_pattern") if cmd.get("fe_pattern") in FE_PATTERNS else meta.get("fe", "banner")
        d["command"] = {
            "effector": eff,
            "category": meta.get("category"),
            "fe_pattern": fe,
            "surface": cmd.get("surface", "on_page") if cmd.get("surface") in SURFACES else "on_page",
            "title": cmd.get("title", "") or "",
            "message": cmd.get("message", "") or "",
            "cta": cmd.get("cta", "") or "",
            "target": cmd.get("target"),
        }
        d["predicted_persona"] = plabel
        # 2) PERSONA-CONDITIONED GATE (leverage the prediction): Franz-silence. Use the stronger of
        #    the heuristic and the coach's own emitted belief. A predicted-Franz user is NOT
        #    interrupted on S3-S5 and is NEVER handed off — budget is saved for the final price.
        belief = d.get("persona_belief") or {}
        franz_like = max(pscores.get("franz", 0.0), float(belief.get("franz", 0) or 0))
        if eff != "NO_ACTION" and franz_like >= 0.6 and (step in _FRANZ_EARLY or eff in _HANDOFF):
            d["command"]["effector"] = "NO_ACTION"
            d["command"]["title"] = d["command"]["message"] = d["command"]["cta"] = ""
            d["reasoning"] = f"persona-gate: franz-silence (franz~{franz_like:.2f} @ {step}, suppressed {eff})"
            d["_acted"] = False
            return d
        if eff != "NO_ACTION":
            self.used += 1
            d["_acted"] = True
        else:
            d["_acted"] = False
        return d

    # ── new public API ──────────────────────────────────────────────────────────────────

    def observe(self, feed: list, step: str) -> dict:
        """Event-feed API: called AFTER EVERY persona micro-turn.

        Takes the raw shared feed (source=user|coach events from the whole session so far),
        filters it to coach-visible events (drops thought/state/feeling; keeps type/target/
        value/t/step from user events + type/target/value/step from coach events), and
        returns either NO_ACTION or ONE effector to inject into the feed.

        The coach does NOT "take the turn" — it is an async OBSERVER that may push ONE
        effector onto the feed that the persona will see on its NEXT turn.
        """
        filtered = self._filter_feed(feed)
        return self.decide(filtered, step)

    @staticmethod
    def _filter_feed(feed: list) -> list:
        """Produce the coach-visible view of the event feed.

        Coach sees:
          - user events: type, step, target, value, t  (NOT thought/state/feeling)
          - coach events: type, step, target, value, t (already stripped)
        Private fields (thought, feeling, state, mental_state, reason, source=coach detail)
        are dropped.
        """
        _VISIBLE = {"type", "step", "target", "value", "t", "source"}
        result = []
        for e in feed:
            if not isinstance(e, dict):
                continue
            result.append({k: e[k] for k in _VISIBLE if k in e})
        return result

    @staticmethod
    def _noop(why: str) -> dict:
        return {"reasoning": why, "command": {"effector": "NO_ACTION", "category": None,
                "fe_pattern": "banner", "surface": "on_page", "title": "", "message": "",
                "cta": "", "target": None}, "hypotheses": [], "value_estimate": 0.0, "_acted": False}
