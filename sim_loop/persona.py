"""LLM persona — system prompt set once at session start, mental state threaded
turn to turn.

NEW (event-feed refactor):
  turn(screen, feed) -> dict with continuation ∈ {more, advance, leave, convert}
    continuation:
      "more"    = hesitating/undecided — watch the feed for coach reaction, act again
      "advance" = proceeding to next funnel screen
      "leave"   = leaving (abandon or service contact)
      "convert" = completing purchase (last screen only)

Backward-compat:
  step(screen) still works; aliases turn(screen).

The persona is NOT locked to one batch per funnel step. It may emit some events
and set continuation="more" to keep acting on the SAME screen (dwell, hesitate,
OR react to a coach widget that just appeared on the feed). It reads the recent
feed via shared_event_feed in the screen (embedded by widget.render each micro-turn).
"""
from __future__ import annotations
import json
from llm import chat, extract_json
from persona_prompt import build_system_prompt

STATE_KEYS = ["attention", "satisfaction", "effort_left", "grasp", "effort_vs_reward"]

# Mapping from internal LLM output values to the canonical continuation enum
_STATUS_TO_CONT = {
    "acting": "more",       # old value → "more"
    "more": "more",         # new value
    "continue": "advance",  # old value → "advance"
    "advance": "advance",   # new value
    "leave": "leave",
    "convert": "convert",
}
_CONT_TO_STATUS = {
    "more": "acting",
    "advance": "continue",
    "leave": "leave",
    "convert": "convert",
}


class LLMPersona:
    def __init__(self, seg: str, session_instance: dict, start_state: dict,
                 model: str | None = None, temperature: float = 0.8):
        self.seg = seg
        self.session_instance = session_instance
        self.system = build_system_prompt(seg, session_instance)
        self.state = dict(start_state)
        self.history_brief: list[str] = []
        self.model = model
        self.temperature = temperature
        self.initial_intent = session_instance.get("visit_goal", "researching")

    # ── new public API ─────────────────────────────────────────────────────────

    def turn(self, screen: dict, feed: list | None = None) -> dict:
        """Event-feed API.

        Parameters
        ----------
        screen : rendered screen dict from widget.render() — already carries
                 shared_event_feed if the orchestrator passed recent_feed.
        feed   : the raw full event feed (append-only list of dicts). If the
                 screen was just re-rendered the feed is already embedded; this
                 parameter lets the persona layer inject truly-just-appended
                 events (e.g. a coach injection that arrived after render).

        Returns dict with:
          events        — list of events emitted this turn
          continuation  — "more" | "advance" | "leave" | "convert"
          status        — backward-compat alias for continuation (old values)
          state, feeling, reason, intent, ...
        """
        # If feed has events not yet in the screen's shared_event_feed, merge them in.
        if feed is not None:
            existing = [e for e in (screen.get("shared_event_feed") or []) if isinstance(e, dict)]
            extra = [e for e in feed[-12:] if isinstance(e, dict) and e not in existing]
            if extra:
                screen = dict(screen)  # shallow copy to avoid mutating
                screen["shared_event_feed"] = (existing + extra)[-12:]

        user = json.dumps(screen, ensure_ascii=False)
        default = {"events": [], "continuation": "leave", "status": "leave",
                   "state": dict(self.state), "feeling": "distracted",
                   "reason": "unparseable", "intent": self.initial_intent}
        try:
            raw = chat([{"role": "system", "content": self.system},
                        {"role": "user", "content": user}],
                       model=self.model, temperature=self.temperature, max_tokens=900)
            out = extract_json(raw)
            if not isinstance(out, dict):
                out = dict(default)
        except Exception:
            out = dict(default)

        # ── normalise continuation / status ───────────────────────────────────
        # Accept either old (status) or new (continuation) field from the LLM output.
        if not isinstance(out.get("events"), list):
            out["events"] = []

        raw_cont = out.get("continuation") or out.get("status")
        cont = _STATUS_TO_CONT.get(str(raw_cont).lower() if raw_cont else "", None)
        if cont is None:
            # fallback: a "decision" field from old prompts
            dec = out.get("decision")
            cont = _STATUS_TO_CONT.get(str(dec).lower() if dec else "", "leave")
        out["continuation"] = cont
        out["status"] = _CONT_TO_STATUS.get(cont, "leave")  # backward compat

        # "decision" field: old-style (continue/leave/convert), None when "more"
        if cont in ("advance", "continue"):
            out["decision"] = "continue"
        elif cont in ("leave", "convert"):
            out["decision"] = cont
        else:
            out["decision"] = None   # "more" = not committed yet

        # ── thread state forward ──────────────────────────────────────────────
        if not isinstance(out.get("state"), dict):
            out["state"] = dict(self.state)
        st = out.get("state", {}) or {}
        new = {}
        for k in STATE_KEYS:
            try:
                new[k] = float(st.get(k, self.state.get(k)))
            except Exception:
                new[k] = self.state.get(k)
        self.state = new
        out["state"] = new

        # ── history brief: only when persona COMMITS (not on a "more" pause) ──
        if cont != "more":
            step = screen.get("you_are_on", "?")
            feel = out.get("feeling", "")
            self.history_brief.append(f"{step}: {out['decision'] or cont}/{feel}")

        return out

    # ── backward-compat alias ─────────────────────────────────────────────────

    def step(self, screen: dict) -> dict:
        """Backward-compatible alias for turn(screen, feed=None).

        Returns the same dict as turn(), so callers that read out.get("status")
        still work via the backward-compat 'status' field.
        """
        return self.turn(screen, feed=None)
