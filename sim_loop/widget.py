"""Widget = deterministic funnel state machine + screen renderer.

It is NOT an LLM. It renders the per-step screen the persona perceives (reusing the
canonical screen templates extracted from datasets/persona_v2), injects any coach
intervention, and advances the funnel.

NEW (event-feed refactor):
  render() now accepts `recent_feed` — the last N events from the shared event feed
  (both user and coach events), embedded in the screen as `shared_event_feed` so
  the persona can react to coach injections that appeared mid-screen.

  Output schema uses `continuation` ∈ {more, advance, leave, convert} instead of
  the old `status` field:
    more    = hesitating / not committed — emit signals, pause, watch the feed
    advance = proceeding to the next screen
    leave   = leaving (abandon or service contact)
    convert = completing purchase (last screen only)
"""
from __future__ import annotations
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATES = json.loads((HERE / "step_templates.json").read_text(encoding="utf-8"))
STEP_ORDER = ["S1_COVERAGE_TYPE", "S2_INSURED_PERSONS", "S3_PERSONAL_INFO",
              "S4_TARIFF_SELECT", "S5_ADDON_SELECT", "S6_PERSONAL_DATA",
              "S7_HEALTH_QUESTIONS", "S8_REVIEW_PURCHASE"]


def first_step() -> str:
    return STEP_ORDER[0]


def next_step(step: str):
    i = STEP_ORDER.index(step)
    return STEP_ORDER[i + 1] if i + 1 < len(STEP_ORDER) else None


def render(step: str, running_state: dict, history_brief: list,
           session_instance: dict, initial_intent: str,
           coach_injection: dict | None = None,
           recent_feed: list | None = None) -> dict:
    """Build the user-turn screen JSON the persona model receives.

    Parameters
    ----------
    step             : current funnel step key
    running_state    : persona mental-state dict
    history_brief    : list of short step summaries so far
    session_instance : session-level instance vars
    initial_intent   : the persona's visit goal
    coach_injection  : optional coach widget command to show on this screen
    recent_feed      : last ~12 events from the SHARED event feed (source=user|coach),
                       embedded as shared_event_feed so the persona sees coach injections
    """
    tpl = TEMPLATES[step]
    screen = json.loads(json.dumps(tpl))  # deep copy
    screen["you_are_on"] = step
    screen["your_running_state"] = running_state
    screen["history_brief"] = history_brief
    screen["session_instance"] = session_instance
    screen["your_initial_intent"] = initial_intent

    # ── PRE-DROPOFF TELEGRAPH: a real user's hesitation is VISIBLE before they leave. Make the
    #    churn-predictive micro-signals legal on every screen, and require the persona to emit
    #    them as its state degrades (so the coach has REAL signals to detect, not a silent bail).
    _asp = screen.setdefault("action_space", {})
    _legal = _asp.setdefault("legal_event_types", [])
    for _ev in ("hover", "price_hover", "cancel_hover", "slow_mouse", "nav_back", "scroll_up",
                "idle", "pause", "tab_blur", "external_nav", "exit_intent", "validation_error",
                "field_clear", "tooltip_open", "rage_click", "text_select", "copy"):
        if _ev not in _legal:
            _legal.append(_ev)
    screen.setdefault("rules", []).append(
        "PRE-DROPOFF TELEGRAPH (important): real hesitation is VISIBLE before a user leaves. As your "
        "state degrades or you lean toward leaving, FIRST emit the observable micro-signals you'd "
        "actually produce, THEN decide — do NOT abandon silently in one clean step. Pick the 1–3 that "
        "fit your feeling: dwell/hover on the price or options WITHOUT selecting; price_hover then "
        "cancel_hover (recoil from the number); nav_back (second-guessing); scroll_up / re-read; "
        "idle/pause (stalling or distracted); tab_blur/external_nav (leaving to compare); "
        "exit_intent (cursor darts to the screen edge / tab bar) right before you abandon; "
        "validation_error/field_clear/text_select/copy when a field or a term frustrates you. "
        "A decisive, satisfied user emits few/none of these and proceeds; only telegraph what you "
        "genuinely feel. Your event trace must SHOW the hesitation building.")

    # ── SHARED EVENT FEED: embed recent events so the persona sees coach injections ──
    if recent_feed:
        screen["shared_event_feed"] = recent_feed   # list of {source,type,step,target,value,t,...}

    # ── OUTPUT SCHEMA: continuation field (replaces old "status") ──────────────
    _osch = screen.get("output_schema")
    if isinstance(_osch, dict):
        _osch["continuation"] = (
            "more | advance | leave | convert — "
            "'more' = you are HESITATING and have NOT committed yet: emit your pre-dropoff "
            "signals and PAUSE (a coach helper may then appear on the shared_event_feed and "
            "you will see it on your NEXT turn on this same screen); "
            "'advance' = proceeding to the next screen; "
            "'leave' = you are leaving (frustrated abandon OR service contact); "
            "'convert' = completing purchase (last screen only). "
            "A decisive flowing user goes straight to 'advance'; 'more' is for GENUINE hesitation only."
        )
        # keep 'status' hint for backward compat with prompts that still emit it
        _osch["status"] = "(deprecated alias for continuation — prefer 'continuation')"

    screen.setdefault("rules", []).append(
        "TURN STATUS — you do NOT have to finish the screen in one turn. Decisive / flowing: emit your "
        "actions and set continuation 'advance' ('convert' only at the final step). Bailing cleanly: 'leave'. "
        "HESITATING (state degrading, not committed): emit your telegraph signals and set continuation "
        "'more' to PAUSE — watch the shared_event_feed; on your NEXT turn you will see any new coach "
        "widget that appeared and can THEN commit. Use 'more' for genuine hesitation, never to stall indefinitely.")

    if coach_injection:
        # a coach surface is shown on this screen; persona may engage or dismiss it
        screen["coach_intervention_shown"] = {
            "surface": coach_injection.get("surface", "on_page"),
            "title": coach_injection.get("title", ""),
            "message": coach_injection.get("message", ""),
            "cta": coach_injection.get("cta", ""),
            "intent": coach_injection.get("effector", ""),
            "category": coach_injection.get("category"),
        }
        asp = screen.get("action_space", {})
        legal = asp.get("legal_event_types", [])
        for ev in ("widget_cta", "widget_dismiss"):
            if ev not in legal:
                legal.append(ev)
        asp["legal_event_types"] = legal
        osch = screen.get("output_schema")
        if isinstance(osch, dict):
            osch["intervention_assessment"] = {
                "engaged": "<true if you USE the helper (also emit a widget_cta event) / false if you "
                           "ignore it (emit widget_dismiss)>",
                "reaction": "helpful | engaging | neutral | distracting | annoying",
                "effect": "<short: did it change your state / decision, and how>",
            }
        screen.setdefault("rules", []).append(
            "A COACH HELPER is on this screen now (`coach_intervention_shown`: intent + title + "
            "message). It does NOT change the funnel — but judge it HONESTLY against the pain you "
            "feel on THIS screen and decide its effect on YOU:\n"
            "• If it directly addresses what is making you hesitate RIGHT NOW — a price you find too "
            "high → a price reframe / breakdown; a long scary form → a 'why + it's short' explainer or "
            "a simplify/skip option; a confusing term → a plain explanation; an unanswered worry → a "
            "checker/FAQ — then it GENUINELY HELPS: emit `widget_cta`, RAISE the relevant state "
            "var(s) (satisfaction / effort_vs_reward / grasp / effort_left) by ~0.15–0.35, and it can "
            "turn a would-be 'leave' into 'continue'. Don't pretend help you didn't need.\n"
            "• If you are SERVICE-AFFINE / overwhelmed and it offers a human path (callback, WhatsApp, "
            "'leave your number and we take it from here', advisor) and you wanted that — TAKE IT: "
            "emit `widget_cta` and set decision 'leave' with a reason that NAMES the callback / "
            "advisor / WhatsApp. That is your PREFERRED outcome (a service contact), NOT a frustrated abandon.\n"
            "• If it is irrelevant, mistimed, pushy, or you didn't need it: emit `widget_dismiss`, "
            "optionally lower attention/satisfaction slightly, and decide as you would have anyway.\n"
            "Fill `intervention_assessment` {reaction, engaged, effect} and let a GOOD, well-timed "
            "match move your decision — a mismatch must NOT save you.")
    return screen
