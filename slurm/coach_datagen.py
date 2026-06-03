"""
Coach distillation datagen: run the distilled persona model through the funnel,
apply the inference-time shim (force "acting" on hesitation), and call the teacher
LLM (OpenRouter) for coach decisions.

Output:
  - sessions_for_coach.jsonl  — full session recordings with coach decisions
  - coach_sft/train.jsonl + val.jsonl — coach SFT pairs ready for training

The SHIM: after persona commits, if events contain hesitation signals AND decision=="leave",
we retroactively treat it as "acting": let the coach observe, inject a widget, then
re-prompt the persona with the widget visible. If persona was going to leave cleanly
(no hesitation), coach never gets called (preserves silence).

Usage (on Leonardo, 1 GPU):
    python slurm/coach_datagen.py --base ~/models/minicpm5-1b \
        --adapter ~/zero-one/leonardo/out_v6_unified \
        --n 60 --batch_size 24 --out ~/zero-one/leonardo/coach_data
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "sim_loop"))

import widget as wdgt
from coach import CoachModel, COACH_SYSTEM, predict_persona

_POOLS_PATH = _REPO / "sim_loop" / "session_pools.json"
POOLS = json.loads(_POOLS_PATH.read_text()) if _POOLS_PATH.exists() else None

PERSONAS = ["judith", "franz", "peter"]
REAL = {"judith": 0.206, "franz": 0.406, "peter": 0.388}
STEPS = list(wdgt.STEP_ORDER)

# The distilled persona model was trained on old format (status/decision). Map to new continuation.
_DECISION_TO_CONT = {"continue": "advance", "leave": "leave", "convert": "convert"}
_STATUS_TO_CONT = {"acting": "more", "more": "more", "continue": "advance",
                   "advance": "advance", "leave": "leave", "convert": "convert"}

# Hesitation signals that trigger the shim
HESITATION_EVENTS = {"hover", "price_hover", "cancel_hover", "slow_mouse", "nav_back",
                     "scroll_up", "idle", "pause", "tab_blur", "external_nav",
                     "exit_intent", "validation_error", "field_clear", "tooltip_open",
                     "rage_click", "text_select", "copy"}

SUCCESS = {
    "judith": {"convert", "advisor_handoff"},
    "franz": {"convert"},
    "peter": {"advisor_handoff", "convert"},
}


def sample_session_instance(rng: random.Random) -> dict:
    if POOLS is None:
        return {}
    p = POOLS["pools"]
    return {k: rng.choice(v) for k, v in p.items()}


def has_hesitation(events: list[dict]) -> bool:
    """Check if persona output contains hesitation signals."""
    return any(str(e.get("type", "")) in HESITATION_EVENTS for e in events if isinstance(e, dict))


def strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def filtered_event(e: dict) -> dict:
    return {k: e.get(k) for k in ("step", "type", "target", "value", "t") if k in e}


def classify_outcome(decision: str, reason: str, last_step: bool) -> str:
    if decision == "continue" and last_step:
        return "convert"
    if decision == "leave":
        r = (reason or "").lower()
        if any(k in r for k in ("advisor", "person", "call", "human", "whatsapp",
                                "phone", "contact", "agent", "berat")):
            return "advisor_handoff"
        return "abandon"
    return "in_progress"


class PersonaInference:
    """Distilled persona model — batched inference with the shim."""

    def __init__(self, base: str, adapter: str, batch_size: int = 24,
                 max_new_tokens: int = 768):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        self.model = PeftModel.from_pretrained(model, adapter)
        self.model.eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def generate_single(self, persona: str, screen: dict) -> dict:
        """Single inference call — returns parsed persona output."""
        msgs = [
            {"role": "system", "content": f"persona: {persona}"},
            {"role": "user", "content": json.dumps(screen, ensure_ascii=False)},
        ]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        with self._torch.no_grad():
            gen = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=0.9, top_p=0.95,
                pad_token_id=self.tok.pad_token_id)
        raw = self.tok.decode(gen[0][plen:], skip_special_tokens=True)
        try:
            return json.loads(strip_fences(raw))
        except Exception:
            return {"events": [], "decision": "leave", "reason": "parse_error"}

    def generate_batch(self, items: list[tuple[str, dict]]) -> list[dict]:
        """Batch inference: items = [(persona, screen), ...]."""
        results = []
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            msgs_list = [
                [{"role": "system", "content": f"persona: {p}"},
                 {"role": "user", "content": json.dumps(s, ensure_ascii=False)}]
                for p, s in chunk
            ]
            texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in msgs_list]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            plen = enc["input_ids"].shape[1]
            with self._torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens,
                    do_sample=True, temperature=0.9, top_p=0.95,
                    pad_token_id=self.tok.pad_token_id)
            for g in gen:
                raw = self.tok.decode(g[plen:], skip_special_tokens=True)
                try:
                    results.append(json.loads(strip_fences(raw)))
                except Exception:
                    results.append({"events": [], "decision": "leave", "reason": "parse_error"})
        return results


def _normalize_continuation(out: dict) -> str:
    """Extract continuation from persona model output (handles old+new formats)."""
    # New format: continuation field
    raw = out.get("continuation") or out.get("status") or out.get("decision")
    if raw:
        raw = str(raw).lower()
    cont = _STATUS_TO_CONT.get(raw) or _DECISION_TO_CONT.get(raw, "leave")
    return cont


def run_session(persona_model: PersonaInference, persona: str, seed: int,
                coach_budget: int = 2, coach_model_name: str | None = None) -> dict:
    """Run one session with the inference-time shim.

    Uses the event-feed architecture:
      - persona produces events per micro-turn
      - coach.observe(feed, step) after each micro-turn
      - shim: if persona hesitates+leaves, retroactively give coach a shot
    """
    rng = random.Random(seed)
    si = sample_session_instance(rng)
    initial_intent = si.get("visit_goal", "researching")

    # Coach = teacher LLM via OpenRouter
    coach = CoachModel(mode="active", model=coach_model_name, budget=coach_budget,
                       temperature=0.4)

    state = {"attention": 1.0, "satisfaction": 0.7, "effort_left": 1.0,
             "grasp": 1.0, "effort_vs_reward": 0.7}
    history_brief: list[str] = []
    feed: list[dict] = []  # shared append-only event feed
    steps_rec: list[dict] = []
    outcome = "abandon"
    coach_observations: list[dict] = []  # for SFT extraction

    step = wdgt.first_step()
    while step is not None:
        last = wdgt.next_step(step) is None

        # === TURN 1: persona acts on the screen (no coach widget yet) ===
        screen1 = wdgt.render(step, dict(state), list(history_brief),
                              si, initial_intent, coach_injection=None,
                              recent_feed=feed[-12:])
        out1 = persona_model.generate_single(persona, screen1)

        # Normalize to new continuation enum
        events1 = out1.get("events", []) if isinstance(out1.get("events"), list) else []
        cont1 = _normalize_continuation(out1)

        # Add events to feed
        for e in events1:
            if isinstance(e, dict):
                e.setdefault("step", step)
                e.setdefault("source", "user")
                feed.append(e)

        # === THE SHIM: if hesitation + leave → force coach observation ===
        coach_dec = None
        out_final = out1
        shim_triggered = False
        final_cont = cont1

        if cont1 == "leave" and has_hesitation(events1) and coach.used < coach.budget \
                and step not in CoachModel.DETECTION_ONLY:
            shim_triggered = True
            # Coach observes the full feed (uses new observe() API)
            coach_dec = coach.observe(feed, step=step)

            # Record coach observation for SFT
            coach_observations.append({
                "step": step,
                "filtered_feed": coach._filter_feed(feed),
                "form_state": None,
                "coach_output": coach_dec,
                "shim_triggered": True,
            })

            if coach_dec.get("_acted"):
                # Coach injected — re-prompt persona with the widget
                injection = coach_dec["command"]
                feed.append({"type": "widget_shown", "step": step, "source": "coach",
                             "t": 0.0, "target": injection.get("effector"),
                             "value": injection.get("effector")})

                screen2 = wdgt.render(step, dict(state), list(history_brief),
                                      si, initial_intent, coach_injection=injection,
                                      recent_feed=feed[-12:])
                out2 = persona_model.generate_single(persona, screen2)

                events2 = out2.get("events", []) if isinstance(out2.get("events"), list) else []
                for e in events2:
                    if isinstance(e, dict):
                        e.setdefault("step", step)
                        e.setdefault("source", "user")
                        feed.append(e)

                final_cont = _normalize_continuation(out2)
                out_final = out2
        elif cont1 != "leave" and step not in CoachModel.DETECTION_ONLY:
            # Even on advance, let coach observe (it may choose NO_ACTION) — for training data
            if coach.used < coach.budget:
                coach_dec = coach.observe(feed, step=step)
                coach_observations.append({
                    "step": step,
                    "filtered_feed": coach._filter_feed(feed),
                    "form_state": None,
                    "coach_output": coach_dec,
                    "shim_triggered": False,
                })

        # Update state from final persona output
        if isinstance(out_final.get("state"), dict):
            for k in state:
                try:
                    state[k] = float(out_final["state"].get(k, state[k]))
                except (TypeError, ValueError):
                    pass

        # Map continuation to old decision for classify_outcome
        decision_compat = "continue" if final_cont == "advance" else final_cont
        history_brief.append(f"{step}: {decision_compat}/{out_final.get('feeling', '')}")

        steps_rec.append({
            "step": step,
            "persona_output": out_final,
            "coach_decision": coach_dec,
            "shim_triggered": shim_triggered,
            "continuation": final_cont,
        })

        # Resolve outcome
        outcome_here = classify_outcome(decision_compat, out_final.get("reason", ""), last)
        if outcome_here != "in_progress":
            outcome = outcome_here
            break
        if last:
            outcome = "convert"
            break
        step = wdgt.next_step(step)

    return {
        "persona": persona,
        "session_instance": si,
        "outcome": outcome,
        "n_steps": len(steps_rec),
        "coach_interventions": coach.used,
        "feed": feed,
        "steps": steps_rec,
        "coach_observations": coach_observations,
    }


def extract_coach_sft(sessions: list[dict]) -> list[dict]:
    """Extract coach SFT pairs from session recordings.

    Each pair: system="coach", user=observation context, assistant=coach JSON output.
    Includes both acted and NO_ACTION decisions (the model must learn WHEN to wait).
    """
    pairs = []
    for sess in sessions:
        for obs in sess.get("coach_observations", []):
            coach_out = obs.get("coach_output", {})
            if not coach_out or coach_out.get("reasoning", "").startswith("detection-only"):
                continue

            # Build the observation context (what the coach saw)
            context = {
                "current_step": obs["step"],
                "activity_log": obs["filtered_feed"],
                "annoyance_budget_left": 2 - sum(
                    1 for o in sess.get("coach_observations", [])
                    if o.get("coach_output", {}).get("_acted") and
                    sess["coach_observations"].index(o) < sess["coach_observations"].index(obs)
                ),
            }

            # The coach output (what the teacher decided)
            # Remove internal fields
            output = {k: v for k, v in coach_out.items() if not k.startswith("_")}

            pairs.append({
                "messages": [
                    {"role": "system", "content": "coach"},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
                ]
            })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model path")
    ap.add_argument("--adapter", required=True, help="Persona LoRA adapter path")
    ap.add_argument("--n", type=int, default=60, help="Sessions per arm")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--coach_budget", type=int, default=2)
    ap.add_argument("--coach_model", default=None, help="OpenRouter model for coach teacher")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="leonardo/coach_data")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading persona model: {args.base} + {args.adapter}", flush=True)
    persona_model = PersonaInference(args.base, args.adapter, args.batch_size)

    # Sample personas with real proportions
    rng = random.Random(args.seed)
    pool = list(REAL.keys())
    wts = [REAL[p] for p in pool]
    assign = [rng.choices(pool, wts)[0] for _ in range(args.n)]

    print(f"Running {args.n} sessions (mix: {Counter(assign)})", flush=True)
    t0 = time.time()

    sessions = []
    for i, persona in enumerate(assign):
        if (i + 1) % 10 == 0:
            print(f"  session {i+1}/{args.n} ({time.time()-t0:.0f}s)", flush=True)
        try:
            sess = run_session(persona_model, persona, args.seed * 1000 + i,
                               coach_budget=args.coach_budget,
                               coach_model_name=args.coach_model)
            sessions.append(sess)
        except Exception as e:
            sys.stderr.write(f"[session {i}] failed: {e}\n")

    dt = time.time() - t0
    print(f"\nDone: {len(sessions)} sessions in {dt:.1f}s", flush=True)

    # Save raw sessions
    sessions_file = out_dir / "sessions_for_coach.jsonl"
    with open(sessions_file, "w") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Extract coach SFT
    pairs = extract_coach_sft(sessions)
    print(f"Coach SFT pairs: {len(pairs)}", flush=True)

    # Split train/val (90/10)
    rng2 = random.Random(args.seed + 999)
    rng2.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.1))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    sft_dir = out_dir / "coach_sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    with open(sft_dir / "train.jsonl", "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(sft_dir / "val.jsonl", "w") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Stats
    outcomes = Counter(s["outcome"] for s in sessions)
    interventions = sum(s["coach_interventions"] for s in sessions)
    shims = sum(1 for s in sessions for st in s["steps"] if st.get("shim_triggered"))
    acted = sum(1 for p in pairs if json.loads(p["messages"][2]["content"]).get("command", {}).get("effector") != "NO_ACTION")

    summary = {
        "n_sessions": len(sessions),
        "outcomes": dict(outcomes),
        "total_coach_interventions": interventions,
        "shim_triggers": shims,
        "coach_sft_pairs": len(pairs),
        "coach_sft_acted": acted,
        "coach_sft_noaction": len(pairs) - acted,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "wall_sec": round(dt, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
