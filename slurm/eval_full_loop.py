"""
Full-loop eval: distilled persona + distilled coach, both as LoRA adapters.

BATCHED: all sessions per arm advance through the funnel in lockstep.
  - off arm: pure batch (no coach), same as persona_traces.py
  - on arm: batch per step, then shim check — only sessions with hesitation+leave
    get individual coach calls + persona re-prompt. The rest advance as a batch.

Usage:
    python slurm/eval_full_loop.py \
        --base ~/models/minicpm5-1b \
        --persona_adapter ~/zero-one/leonardo/out_v6_unified \
        --coach_adapter ~/zero-one/leonardo/out_coach_lora \
        --n 100 --batch_size 48
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "sim_loop"))

import widget as wdgt
from coach import _BY_ID

PERSONAS = ["judith", "franz", "peter"]
STEPS = list(wdgt.STEP_ORDER)
STATE_KEYS = ["attention", "satisfaction", "effort_left", "grasp", "effort_vs_reward"]

HESITATION_EVENTS = {"hover", "price_hover", "cancel_hover", "slow_mouse", "nav_back",
                     "scroll_up", "idle", "pause", "tab_blur", "external_nav",
                     "exit_intent", "validation_error", "field_clear", "tooltip_open",
                     "rage_click", "text_select", "copy"}
DETECTION_ONLY = {"S1_COVERAGE_TYPE", "S2_INSURED_PERSONS"}

SUCCESS = {
    "judith": {"convert", "advisor_handoff"},
    "franz": {"convert"},
    "peter": {"advisor_handoff", "convert"},
}

_DECISION_TO_CONT = {"continue": "advance", "leave": "leave", "convert": "convert"}
# "more" and "acting" = hesitation from multi-turn persona; distilled model can't
# actually yield mid-screen, so treat both as advance (session proceeds)
_STATUS_TO_CONT = {"acting": "advance", "more": "advance", "continue": "advance",
                   "advance": "advance", "leave": "leave", "convert": "convert"}

_POOLS_PATH = _REPO / "sim_loop" / "session_pools.json"
POOLS = json.loads(_POOLS_PATH.read_text()) if _POOLS_PATH.exists() else None


def sample_session_instance(rng: random.Random) -> dict:
    if POOLS is None:
        return {}
    p = POOLS["pools"]
    return {k: rng.choice(v) for k, v in p.items()}


def strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def filtered_event(e: dict) -> dict:
    return {k: e.get(k) for k in ("step", "type", "target", "value", "t", "source") if k in e}


def has_hesitation(events: list[dict]) -> bool:
    return any(str(e.get("type", "")) in HESITATION_EVENTS for e in events if isinstance(e, dict))


def _normalize_continuation(out: dict) -> str:
    raw = out.get("continuation") or out.get("status") or out.get("decision")
    if raw:
        raw = str(raw).lower()
    return _STATUS_TO_CONT.get(raw) or _DECISION_TO_CONT.get(raw, "leave")


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


class DualModel:
    """Loads persona + coach LoRA adapters. Supports batched and single inference."""

    def __init__(self, base: str, persona_adapter: str, coach_adapter: str | None = None,
                 max_new_tokens: int = 768, batch_size: int = 48):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.persona_model = PeftModel.from_pretrained(base_model, persona_adapter)
        self.persona_model.eval()

        self.coach_model = None
        if coach_adapter:
            base_model2 = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            self.coach_model = PeftModel.from_pretrained(base_model2, coach_adapter)
            self.coach_model.eval()

    def _batch_generate(self, model, msgs_list: list[list[dict]]) -> list[str]:
        results = []
        for i in range(0, len(msgs_list), self.batch_size):
            chunk = msgs_list[i:i + self.batch_size]
            texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            plen = enc["input_ids"].shape[1]
            with self._torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=self.max_new_tokens,
                    do_sample=True, temperature=0.9, top_p=0.95,
                    pad_token_id=self.tok.pad_token_id)
            results.extend([self.tok.decode(g[plen:], skip_special_tokens=True) for g in gen])
        return results

    def _single_generate(self, model, msgs: list[dict]) -> str:
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        with self._torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=0.9, top_p=0.95,
                pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(gen[0][plen:], skip_special_tokens=True)

    def persona_batch(self, items: list[tuple[str, dict]]) -> list[dict]:
        """Batch persona inference. items = [(persona_name, screen), ...]"""
        msgs_list = [
            [{"role": "system", "content": f"persona: {p}"},
             {"role": "user", "content": json.dumps(s, ensure_ascii=False)}]
            for p, s in items
        ]
        raws = self._batch_generate(self.persona_model, msgs_list)
        results = []
        for raw in raws:
            try:
                results.append(json.loads(strip_fences(raw)))
            except Exception:
                results.append({"events": [], "decision": "leave", "reason": "parse_error"})
        return results

    def persona_single(self, persona: str, screen: dict) -> dict:
        msgs = [
            {"role": "system", "content": f"persona: {persona}"},
            {"role": "user", "content": json.dumps(screen, ensure_ascii=False)},
        ]
        raw = self._single_generate(self.persona_model, msgs)
        try:
            return json.loads(strip_fences(raw))
        except Exception:
            return {"events": [], "decision": "leave", "reason": "parse_error"}

    def coach_single(self, context: dict) -> dict:
        if self.coach_model is None:
            return {"command": {"effector": "NO_ACTION"}, "_acted": False}
        msgs = [
            {"role": "system", "content": "coach"},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        raw = self._single_generate(self.coach_model, msgs)
        try:
            d = json.loads(strip_fences(raw))
        except Exception:
            return {"command": {"effector": "NO_ACTION"}, "_acted": False}
        cmd = d.get("command", {}) or {}
        eff = cmd.get("effector", "NO_ACTION")
        if eff not in _BY_ID and eff != "NO_ACTION":
            eff = "NO_ACTION"
        d["_acted"] = eff != "NO_ACTION"
        return d

    def coach_batch(self, contexts: list[dict]) -> list[dict]:
        """Batch coach inference."""
        if self.coach_model is None:
            return [{"command": {"effector": "NO_ACTION"}, "_acted": False}] * len(contexts)
        msgs_list = [
            [{"role": "system", "content": "coach"},
             {"role": "user", "content": json.dumps(c, ensure_ascii=False)}]
            for c in contexts
        ]
        raws = self._batch_generate(self.coach_model, msgs_list)
        results = []
        for raw in raws:
            try:
                d = json.loads(strip_fences(raw))
            except Exception:
                d = {}
            cmd = d.get("command", {}) or {}
            eff = cmd.get("effector", "NO_ACTION")
            if eff not in _BY_ID and eff != "NO_ACTION":
                eff = "NO_ACTION"
            d["_acted"] = eff != "NO_ACTION"
            results.append(d)
        return results


def run_arm_batched(dual: DualModel, persona: str, n: int, seed: int,
                    arm: str = "off", coach_budget: int = 2) -> list[dict]:
    """Run N sessions for one persona in one arm, batched through the funnel."""
    # Init sessions
    sessions = []
    for i in range(n):
        rng = random.Random(seed + i)
        si = sample_session_instance(rng)
        sessions.append({
            "rng": rng, "si": si,
            "initial_intent": si.get("visit_goal", "researching"),
            "state": dict(zip(STATE_KEYS, [1.0, 0.7, 1.0, 1.0, 0.7])),
            "history_brief": [],
            "feed": [],
            "active": True,
            "outcome": "abandon",
            "n_steps": 0,
            "coach_used": 0,
            "coach_interventions": 0,
            "steps_detail": [],
        })

    for step_idx, step_name in enumerate(STEPS):
        last = step_idx == len(STEPS) - 1
        active = [(i, s) for i, s in enumerate(sessions) if s["active"]]
        if not active:
            break

        # === BATCH: all active personas generate at once ===
        items = []
        for _, s in active:
            screen = wdgt.render(
                step_name, dict(s["state"]), list(s["history_brief"]),
                s["si"], s["initial_intent"],
                coach_injection=None, recent_feed=s["feed"][-12:])
            items.append((persona, screen))

        outs = dual.persona_batch(items)

        # === Process outputs, identify shim candidates ===
        shim_candidates = []  # (index_in_active, session, events, out)

        for (idx, s), out in zip(active, outs):
            events = out.get("events", []) if isinstance(out.get("events"), list) else []
            cont = _normalize_continuation(out)

            for e in events:
                if isinstance(e, dict):
                    e.setdefault("step", step_name)
                    e.setdefault("source", "user")
                    s["feed"].append(e)

            # Update state
            if isinstance(out.get("state"), dict):
                for k in s["state"]:
                    try:
                        s["state"][k] = float(out["state"].get(k, s["state"][k]))
                    except (TypeError, ValueError):
                        pass

            s["_cont"] = cont
            s["_out"] = out
            s["_events"] = events
            s["_coach_acted"] = False

            # Shim candidate?
            if arm == "on" and cont == "leave" and has_hesitation(events) \
                    and s["coach_used"] < coach_budget and step_name not in DETECTION_ONLY:
                shim_candidates.append((idx, s, events, out))

        # === BATCH COACH for shim candidates ===
        if shim_candidates and dual.coach_model is not None:
            contexts = []
            for _, s, _, _ in shim_candidates:
                contexts.append({
                    "current_step": step_name,
                    "activity_log": [filtered_event(e) for e in s["feed"]][-20:],
                    "annoyance_budget_left": coach_budget - s["coach_used"],
                })
            coach_decs = dual.coach_batch(contexts)

            # For sessions where coach acted: batch re-prompt persona
            reprompt_items = []  # (shim_idx, session, injection)
            for (orig_idx, s, events, out), cdec in zip(shim_candidates, coach_decs):
                if cdec.get("_acted"):
                    s["coach_used"] += 1
                    injection = cdec.get("command", {})
                    s["feed"].append({"type": "widget_shown", "step": step_name,
                                      "source": "coach", "t": 0.0,
                                      "target": injection.get("effector")})
                    reprompt_items.append((orig_idx, s, injection))

            if reprompt_items:
                # Batch persona re-prompt
                re_items = []
                for _, s, injection in reprompt_items:
                    screen2 = wdgt.render(
                        step_name, dict(s["state"]), list(s["history_brief"]),
                        s["si"], s["initial_intent"],
                        coach_injection=injection, recent_feed=s["feed"][-12:])
                    re_items.append((persona, screen2))

                re_outs = dual.persona_batch(re_items)

                for (_, s, injection), out2 in zip(reprompt_items, re_outs):
                    events2 = out2.get("events", []) if isinstance(out2.get("events"), list) else []
                    for e in events2:
                        if isinstance(e, dict):
                            e.setdefault("step", step_name)
                            e.setdefault("source", "user")
                            s["feed"].append(e)
                    if isinstance(out2.get("state"), dict):
                        for k in s["state"]:
                            try:
                                s["state"][k] = float(out2["state"].get(k, s["state"][k]))
                            except (TypeError, ValueError):
                                pass
                    s["_cont"] = _normalize_continuation(out2)
                    s["_out"] = out2
                    s["_coach_acted"] = True

        # === Also batch coach for non-leave active sessions (on arm) for observational data ===
        if arm == "on" and dual.coach_model is not None:
            observe_candidates = []
            for (idx, s), out in zip(active, outs):
                if s["_cont"] != "leave" and step_name not in DETECTION_ONLY \
                        and s["coach_used"] < coach_budget and (idx, s, None, None) not in shim_candidates:
                    observe_candidates.append((idx, s))
            if observe_candidates:
                obs_contexts = [{
                    "current_step": step_name,
                    "activity_log": [filtered_event(e) for e in s["feed"]][-20:],
                    "annoyance_budget_left": coach_budget - s["coach_used"],
                } for _, s in observe_candidates]
                # Fire-and-forget: coach observes but we don't use the result
                # (no widget shown since persona is advancing). Just for completeness.
                # Skip this to save time — coach only matters on shim triggers.

        # === Finalize all sessions for this step ===
        for _, s in active:
            cont = s["_cont"]
            out = s["_out"]
            decision_compat = "continue" if cont == "advance" else cont
            s["history_brief"].append(f"{step_name}: {decision_compat}/{out.get('feeling', '')}")
            s["n_steps"] += 1
            s["steps_detail"].append({
                "step": step_name,
                "continuation": cont,
                "coach_acted": s["_coach_acted"],
            })

            outcome_here = classify_outcome(decision_compat, out.get("reason", ""), last)
            if outcome_here != "in_progress":
                s["outcome"] = outcome_here
                s["active"] = False
            elif last:
                s["outcome"] = "convert"
                s["active"] = False

    # Survivors convert
    for s in sessions:
        if s["active"]:
            s["outcome"] = "convert"

    return [{
        "persona": persona,
        "arm": arm,
        "outcome": s["outcome"],
        "n_steps": s["n_steps"],
        "coach_interventions": s["coach_used"],
        "steps": s["steps_detail"],
    } for s in sessions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--persona_adapter", required=True)
    ap.add_argument("--coach_adapter", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--coach_budget", type=int, default=2)
    ap.add_argument("--seed", type=int, default=800)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path(args.coach_adapter or "leonardo") / "eval_full_loop.json"

    print(f"Loading dual model: {args.base}", flush=True)
    print(f"  persona: {args.persona_adapter}", flush=True)
    print(f"  coach: {args.coach_adapter}", flush=True)
    dual = DualModel(args.base, args.persona_adapter, args.coach_adapter,
                     batch_size=args.batch_size)

    all_results = {}
    total_t0 = time.time()

    for arm in ["off", "on"]:
        arm_results = {}
        for persona in PERSONAS:
            t0 = time.time()
            print(f"\n[{arm}/{persona}] running {args.n} sessions (batched) ...", flush=True)
            sessions = run_arm_batched(
                dual, persona, args.n,
                seed=args.seed + (1000 if arm == "on" else 0),
                arm=arm, coach_budget=args.coach_budget)
            dt = time.time() - t0

            outcomes = Counter(s["outcome"] for s in sessions)
            success = sum(1 for s in sessions if s["outcome"] in SUCCESS[persona])
            interventions = sum(s["coach_interventions"] for s in sessions)
            avg_steps = sum(s["n_steps"] for s in sessions) / len(sessions)
            shims = sum(1 for s in sessions for st in s["steps"] if st.get("coach_acted"))

            stats = {
                "n": len(sessions),
                "outcomes": dict(outcomes),
                "success": success,
                "success_rate": round(success / len(sessions), 3),
                "convert_rate": round(outcomes.get("convert", 0) / len(sessions), 3),
                "abandon_rate": round(outcomes.get("abandon", 0) / len(sessions), 3),
                "advisor_rate": round(outcomes.get("advisor_handoff", 0) / len(sessions), 3),
                "avg_steps": round(avg_steps, 2),
                "coach_interventions": interventions,
                "shim_triggers": shims,
                "wall_sec": round(dt, 1),
            }
            arm_results[persona] = stats
            print(f"[{arm}/{persona}] {dt:.1f}s | success={stats['success_rate']} "
                  f"convert={stats['convert_rate']} abandon={stats['abandon_rate']} "
                  f"advisor={stats['advisor_rate']} interventions={interventions}", flush=True)

        all_results[arm] = arm_results

    total_dt = time.time() - total_t0

    # Build report
    report = {
        "model": args.base,
        "persona_adapter": args.persona_adapter,
        "coach_adapter": args.coach_adapter,
        "n_per_persona_per_arm": args.n,
        "total_sessions": args.n * 3 * 2,
        "total_wall_sec": round(total_dt, 1),
        "arms": {},
    }

    for arm in ["off", "on"]:
        arm_data = all_results[arm]
        total_n = sum(v["n"] for v in arm_data.values())
        total_success = sum(v["success"] for v in arm_data.values())
        total_convert = sum(v["outcomes"].get("convert", 0) for v in arm_data.values())
        total_interventions = sum(v["coach_interventions"] for v in arm_data.values())
        report["arms"][arm] = {
            "personas": arm_data,
            "overall_success_rate": round(total_success / total_n, 3),
            "overall_convert_rate": round(total_convert / total_n, 3),
            "total_interventions": total_interventions,
        }

    off_success = report["arms"]["off"]["overall_success_rate"]
    on_success = report["arms"]["on"]["overall_success_rate"]
    off_convert = report["arms"]["off"]["overall_convert_rate"]
    on_convert = report["arms"]["on"]["overall_convert_rate"]

    report["uplift"] = {
        "success_rate_delta": round(on_success - off_success, 3),
        "convert_rate_delta": round(on_convert - off_convert, 3),
        "success_rate_relative": round((on_success - off_success) / max(off_success, 0.001), 3),
        "per_persona": {},
    }
    for p in PERSONAS:
        off_s = all_results["off"][p]["success_rate"]
        on_s = all_results["on"][p]["success_rate"]
        report["uplift"]["per_persona"][p] = {
            "off": off_s, "on": on_s, "delta": round(on_s - off_s, 3),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"RESULTS: coach_off success={off_success} | coach_on success={on_success}")
    print(f"UPLIFT: success {on_success - off_success:+.3f} | convert {on_convert - off_convert:+.3f}")
    for p in PERSONAS:
        u = report["uplift"]["per_persona"][p]
        print(f"  {p}: {u['off']} → {u['on']} ({u['delta']:+.3f})")
    print(f"Total: {args.n * 6} sessions in {total_dt:.1f}s")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
