"""Run the persona <-> coach loop over a SHARED, append-only EVENT FEED.

Architecture (event-feed refactor):
  - Shared event feed (append-only list per session). Every event carries a
    `source` ("user" = persona, "coach" = injected effector).
  - Persona = a multi-turn event PRODUCER. Each micro-turn:
      screen = widget.render(..., recent_feed=...)
      out = persona.turn(screen, feed)    # continuation ∈ {more,advance,leave,convert}
  - Coach = an async OBSERVER. Called AFTER EVERY persona micro-turn:
      obs = coach.observe(feed, step)     # returns NO_ACTION | one effector
    If it acts, the effector is appended to the feed (source=coach) and the
    persona sees it on its NEXT turn on the same screen.
  - Orchestrator interleave:
      render → persona.turn → append events → coach.observe → if acted, append
      coach event → if continuation=="more": persona acts again on same screen
      (now seeing the coach injection); on "advance" → next funnel step;
      "leave"/"convert" → terminate.
  - Per-screen MICRO-TURN CAP (MAX_MICRO=4) prevents infinite dwell.
  - Total-turn cap (MAX_TOTAL_TURNS) prevents runaway sessions.

Two output arms:
  sessions_coach_off.jsonl  — coach ALWAYS skips (control)
  sessions_coach_on.jsonl   — coach is active (LLM policy, annoyance budget)

Session record schema:
  {
    persona, arm, session_instance, outcome, n_steps,
    coach_interventions,
    feed: [{source, type, step, target, value, t, ...}],  # full event feed
    micro_turns: [{step, turn_idx, events, continuation, coach_obs}],  # per-turn
    steps: [...]   # backward-compat steps-shaped view (same as before)
  }

Usage:
  python sim_loop/run.py --sessions 30 --proportions real --out sim_loop/out
  python sim_loop/run.py --sessions 6 --arms off,on --coach-budget 2 --concurrency 4
"""
from __future__ import annotations
import argparse, json, os, random, sys, pathlib, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import widget
from persona import LLMPersona
from coach import CoachModel

POOLS = json.loads((pathlib.Path(__file__).resolve().parent / "session_pools.json").read_text())
PERSONAS = ["judith", "franz", "peter"]
# normalized real segment shares (segment_share 0.155/0.305/0.292 -> renormalized)
REAL = {"judith": 0.206, "franz": 0.406, "peter": 0.388}
BALANCED = {"judith": 1 / 3, "franz": 1 / 3, "peter": 1 / 3}

MAX_MICRO = 4          # persona micro-turns per screen before forced advance
MAX_TOTAL_TURNS = 40   # total micro-turns across the whole session (safety cap)


def sample_session_instance(rng: random.Random) -> dict:
    p = POOLS["pools"]
    return {k: rng.choice(v) for k, v in p.items()}


# Persona-dependent CONVERSION (the formalization's reward ρ): the RIGHT outcome differs by
# segment. Judith: online OR a clean advisor handoff. Franz: online ONLY (advisor = failure).
# Peter: a qualified service contact (callback / WhatsApp / phone) — his handoff IS his conversion.
SUCCESS = {
    "judith": {"convert", "advisor_handoff"},
    "franz": {"convert"},
    "peter": {"advisor_handoff", "convert"},
}


def is_success(persona: str, outcome: str) -> bool:
    return outcome in SUCCESS.get(persona, {"convert"})


def classify_outcome(decision: str, reason: str, last_step: bool) -> str:
    if decision == "continue" and last_step:
        return "convert"
    if decision == "leave":
        r = (reason or "").lower()
        if any(k in r for k in ("advisor", "person", "call", "human", "whatsapp",
                                "phone", "contact", "agent", "berat")):
            return "advisor_handoff"   # = a service contact / human handoff
        return "abandon"
    return "in_progress"


def run_session(seg: str, arm: str, model: str, coach_budget: int, seed: int,
                temperature: float = 0.8, coach_system: str | None = None,
                coach_temperature: float | None = None) -> dict:
    rng = random.Random(seed)
    si = sample_session_instance(rng)
    persona = LLMPersona(seg, si, POOLS["start_state"], model=model,
                         temperature=temperature)
    coach = CoachModel(mode=("active" if arm == "on" else "skip"),
                       model=model, budget=coach_budget, system=coach_system,
                       temperature=(coach_temperature if coach_temperature is not None else 0.4))

    # ── SHARED APPEND-ONLY EVENT FEED ───────────────────────────────────────
    feed: list[dict] = []   # every event: source=user|coach, type, step, target, value, t, ...

    # ── OUTPUT STRUCTURES ────────────────────────────────────────────────────
    steps_rec: list[dict] = []      # backward-compat steps-shaped view
    micro_turns_rec: list[dict] = []  # per-micro-turn details (new)

    outcome = "abandon"
    total_turns = 0
    step = widget.first_step()

    while step is not None and total_turns < MAX_TOTAL_TURNS:
        last = widget.next_step(step) is None
        shown: dict | None = None        # coach widget currently on this screen (if any)
        coach_dec_acted: dict | None = None
        step_events: list[dict] = []
        step_micro_turns: list[dict] = []
        continuation = "leave"           # default if cap forces exit

        for turn_idx in range(MAX_MICRO):
            total_turns += 1

            # ── 1. Render screen with fresh recent feed ──────────────────────
            recent = feed[-12:]   # last 12 feed events (already contain source)
            screen = widget.render(
                step,
                persona.state,
                list(persona.history_brief),
                si,
                persona.initial_intent,
                coach_injection=shown,
                recent_feed=recent,
            )

            # ── 2. Persona takes one micro-turn ──────────────────────────────
            out = persona.turn(screen, feed)

            # Stamp each emitted event with step + source=user and push to feed
            turn_events: list[dict] = []
            for e in (out.get("events") or []):
                if isinstance(e, dict):
                    e.setdefault("step", step)
                    e.setdefault("source", "user")
                    feed.append(e)
                    step_events.append(e)
                    turn_events.append(e)

            continuation = out.get("continuation", "leave")

            # ── 3. Coach observes the live feed (AFTER persona micro-turn) ───
            coach_obs: dict | None = None
            if arm == "on":
                coach_obs = coach.observe(feed, step=step)
                if coach_obs.get("_acted"):
                    coach_dec_acted = coach_obs
                    shown = coach_obs["command"]
                    # Append coach effector to feed as source=coach
                    feed.append({
                        "source": "coach",
                        "type": "widget_shown",
                        "step": step,
                        "target": coach_obs["command"].get("effector"),
                        "value": coach_obs["command"].get("effector"),
                        "t": 0.0,
                    })

            # Record this micro-turn
            step_micro_turns.append({
                "step": step,
                "turn_idx": turn_idx,
                "events": turn_events,
                "continuation": continuation,
                "coach_obs": coach_obs,
            })
            micro_turns_rec.append(step_micro_turns[-1])

            # ── 4. Check continuation ────────────────────────────────────────
            if continuation != "more":
                break   # persona COMMITTED: advance / leave / convert
            # continuation == "more" → persona loops, now seeing any new coach event

        # ── Hit micro-turn cap still undecided → force advance ───────────────
        if continuation == "more":
            continuation = "advance"

        out_final = dict(out or {})
        out_final["events"] = step_events
        out_final["continuation"] = continuation
        # backward-compat decision/status fields
        out_final["decision"] = (
            "continue" if continuation == "advance"
            else continuation   # "leave" or "convert"
        )
        out_final["status"] = {
            "advance": "continue", "leave": "leave", "convert": "convert",
        }.get(continuation, "continue")

        steps_rec.append({
            "step": step,
            "shown_coach": shown,
            "persona_output": out_final,
            "coach_decision": coach_dec_acted,
        })

        # ── Session termination logic ────────────────────────────────────────
        if continuation == "leave":
            outcome = classify_outcome("leave", out_final.get("reason", ""), last)
            break
        if continuation == "convert" or last:
            outcome = "convert"
            break
        step = widget.next_step(step)

    return {
        "persona": seg,
        "arm": arm,
        "session_instance": si,
        "outcome": outcome,
        "n_steps": len(steps_rec),
        "coach_interventions": coach.used,
        # ── NEW: full event feed with source ──
        "feed": feed,
        # ── NEW: per-micro-turn details ───────
        "micro_turns": micro_turns_rec,
        # ── BACKWARD COMPAT: steps-shaped view ─
        "steps": steps_rec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20, help="sessions PER ARM")
    ap.add_argument("--arms", default="off,on")
    ap.add_argument("--proportions", choices=["real", "balanced"], default="real")
    ap.add_argument("--coach-budget", type=int, default=2)
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sim_loop/out")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    weights = REAL if args.proportions == "real" else BALANCED
    rng = random.Random(args.seed)
    pool = list(weights.keys())
    wts = [weights[p] for p in pool]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pre-sample the persona assignment so both arms share the SAME persona mix
    assign = [rng.choices(pool, wts)[0] for _ in range(args.sessions)]

    summary = {}
    for arm in arms:
        fname = out_dir / f"sessions_coach_{'off' if arm == 'off' else 'on'}.jsonl"
        jobs = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for i, seg in enumerate(assign):
                jobs.append(ex.submit(run_session, seg, arm, args.model,
                                      args.coach_budget, args.seed * 100000 + i))
            results = []
            for fut in as_completed(jobs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    sys.stderr.write(f"[session] failed: {e}\n")
        with open(fname, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # arm summary
        n = len(results)
        conv = sum(1 for r in results if r["outcome"] == "convert")
        adv = sum(1 for r in results if r["outcome"] == "advisor_handoff")
        ab = sum(1 for r in results if r["outcome"] == "abandon")
        succ = sum(1 for r in results if is_success(r["persona"], r["outcome"]))
        interv = sum(r["coach_interventions"] for r in results)
        summary[arm] = {"file": str(fname), "n": n,
                        "convert": conv, "convert_rate": round(conv / n, 3) if n else 0,
                        "success": succ, "success_rate": round(succ / n, 3) if n else 0,
                        "advisor": adv, "abandon": ab, "coach_interventions": interv}
        print(f"[arm {arm}] n={n} convert={conv} ({summary[arm]['convert_rate']}) "
              f"success={succ} ({summary[arm]['success_rate']}) "
              f"advisor={adv} abandon={ab} interventions={interv} -> {fname}")

    (out_dir / "summary.json").write_text(json.dumps({
        "args": vars(args), "persona_mix": dict(zip(*[pool, [assign.count(p) for p in pool]])),
        "arms": summary,
    }, ensure_ascii=False, indent=2))
    if "off" in summary and "on" in summary:
        d = summary["on"]["convert_rate"] - summary["off"]["convert_rate"]
        ds = summary["on"]["success_rate"] - summary["off"]["success_rate"]
        print(f"\nUPLIFT (on - off): online convert {d:+.3f}  |  persona-success {ds:+.3f}  "
              f"(coach interventions: {summary['on']['coach_interventions']})")
    print(f"summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
