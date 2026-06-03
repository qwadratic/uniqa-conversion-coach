"""
Step 1 of split pipeline: run distilled persona through the funnel on GPU,
save raw traces with events + hesitation signals. NO internet needed.

BATCHED: all sessions advance through the funnel in lockstep. At each step,
all active sessions are batch-generated together → full GPU utilization.
(Same approach as eval_unified.py: 300 sessions in ~18 min.)

Output: persona_traces.jsonl — one line per session with:
  {persona, session_instance, steps: [{step, events, continuation, state, feeling, reason}], feed}

Usage (Leonardo, 1 GPU):
    python slurm/persona_traces.py --base ~/models/minicpm5-1b \
        --adapter ~/zero-one/leonardo/out_v6_unified \
        --n 80 --batch_size 48 --out ~/zero-one/leonardo/coach_data
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

_POOLS_PATH = _REPO / "sim_loop" / "session_pools.json"
POOLS = json.loads(_POOLS_PATH.read_text()) if _POOLS_PATH.exists() else None

PERSONAS = ["judith", "franz", "peter"]
REAL = {"judith": 0.206, "franz": 0.406, "peter": 0.388}
STEPS = list(wdgt.STEP_ORDER)

HESITATION_EVENTS = {"hover", "price_hover", "cancel_hover", "slow_mouse", "nav_back",
                     "scroll_up", "idle", "pause", "tab_blur", "external_nav",
                     "exit_intent", "validation_error", "field_clear", "tooltip_open",
                     "rage_click", "text_select", "copy"}

DETECTION_ONLY = {"S1_COVERAGE_TYPE", "S2_INSURED_PERSONS"}

_DECISION_TO_CONT = {"continue": "advance", "leave": "leave", "convert": "convert"}
# "more" and "acting" both mean hesitation — for a model trained without multi-turn
# support, treat both as advance (the model can't actually yield, so it's noise)
_STATUS_TO_CONT = {"acting": "advance", "more": "advance", "continue": "advance",
                   "advance": "advance", "leave": "leave", "convert": "convert"}

STATE_KEYS = ["attention", "satisfaction", "effort_left", "grasp", "effort_vs_reward"]


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


class BatchedPersona:
    """Batched persona inference — all active sessions at once."""

    def __init__(self, base: str, adapter: str, batch_size: int = 48,
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

    def generate_batch(self, msgs_list: list[list[dict]]) -> list[str]:
        """Batch generate from message lists. Returns raw strings."""
        results = []
        for i in range(0, len(msgs_list), self.batch_size):
            chunk = msgs_list[i:i + self.batch_size]
            texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            plen = enc["input_ids"].shape[1]
            with self._torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens,
                    do_sample=True, temperature=0.9, top_p=0.95,
                    pad_token_id=self.tok.pad_token_id)
            results.extend([self.tok.decode(g[plen:], skip_special_tokens=True) for g in gen])
        return results


def run_all_sessions(model: BatchedPersona, assignments: list[str],
                     seed: int = 500) -> list[dict]:
    """Run all sessions through the funnel in lockstep, batching per step."""
    n = len(assignments)

    # Init sessions
    sessions = []
    for i, persona in enumerate(assignments):
        rng = random.Random(seed * 1000 + i)
        si = sample_session_instance(rng)
        sessions.append({
            "persona": persona,
            "session_instance": si,
            "initial_intent": si.get("visit_goal", "researching"),
            "state": {k: v for k, v in zip(STATE_KEYS, [1.0, 0.7, 1.0, 1.0, 0.7])},
            "history_brief": [],
            "feed": [],
            "steps_rec": [],
            "active": True,
            "outcome": "abandon",
        })

    for step_idx, step_name in enumerate(STEPS):
        last = step_idx == len(STEPS) - 1
        active = [(i, s) for i, s in enumerate(sessions) if s["active"]]
        if not active:
            break

        # Build prompts for all active sessions
        msgs_list = []
        for _, s in active:
            screen = wdgt.render(
                step_name, dict(s["state"]), list(s["history_brief"]),
                s["session_instance"], s["initial_intent"],
                coach_injection=None,
                recent_feed=s["feed"][-12:],
            )
            msgs_list.append([
                {"role": "system", "content": f"persona: {s['persona']}"},
                {"role": "user", "content": json.dumps(screen, ensure_ascii=False)},
            ])

        # Batch generate
        t0 = time.time()
        raws = model.generate_batch(msgs_list)
        dt = time.time() - t0
        print(f"  {step_name}: {len(active)} active, batch in {dt:.1f}s", flush=True)

        # Parse and update
        for (idx, s), raw in zip(active, raws):
            try:
                out = json.loads(strip_fences(raw))
            except Exception:
                out = {}
            if not isinstance(out, dict):
                out = {}

            events = out.get("events", []) if isinstance(out.get("events"), list) else []
            cont = _normalize_continuation(out)

            # Append events to feed
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

            decision_compat = "continue" if cont == "advance" else cont
            s["history_brief"].append(f"{step_name}: {decision_compat}/{out.get('feeling', '')}")

            shim_eligible = (cont == "leave" and has_hesitation(events)
                             and step_name not in DETECTION_ONLY)

            s["steps_rec"].append({
                "step": step_name,
                "events": events,
                "continuation": cont,
                "state": dict(s["state"]),
                "feeling": out.get("feeling", ""),
                "reason": out.get("reason", ""),
                "shim_eligible": shim_eligible,
                "raw_output": out,  # keep full output for coach context
            })

            # Outcome
            outcome_here = classify_outcome(decision_compat, out.get("reason", ""), last)
            if outcome_here != "in_progress":
                s["outcome"] = outcome_here
                s["active"] = False
            elif last:
                s["outcome"] = "convert"
                s["active"] = False

    # Build final results
    results = []
    for s in sessions:
        # Survivors convert
        if s["active"]:
            s["outcome"] = "convert"
        results.append({
            "persona": s["persona"],
            "session_instance": s["session_instance"],
            "outcome": s["outcome"],
            "n_steps": len(s["steps_rec"]),
            "feed": s["feed"],
            "steps": s["steps_rec"],
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--out", default="leonardo/coach_data")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading persona model: {args.base} + {args.adapter}", flush=True)
    model = BatchedPersona(args.base, args.adapter, args.batch_size)

    rng = random.Random(args.seed)
    pool = list(REAL.keys())
    wts = [REAL[p] for p in pool]
    assign = [rng.choices(pool, wts)[0] for _ in range(args.n)]

    print(f"Running {args.n} sessions BATCHED (mix: {Counter(assign)})", flush=True)
    t0 = time.time()

    sessions = run_all_sessions(model, assign, seed=args.seed)

    dt = time.time() - t0

    out_file = out_dir / "persona_traces.jsonl"
    with open(out_file, "w") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    outcomes = Counter(s["outcome"] for s in sessions)
    shim_eligible = sum(1 for s in sessions for st in s["steps"] if st.get("shim_eligible"))
    summary = {
        "n_sessions": len(sessions),
        "outcomes": dict(outcomes),
        "shim_eligible_steps": shim_eligible,
        "wall_sec": round(dt, 1),
        "sessions_per_sec": round(len(sessions) / dt, 2),
        "persona_mix": dict(Counter(assign)),
    }
    (out_dir / "traces_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone: {len(sessions)} sessions in {dt:.1f}s ({summary['sessions_per_sec']} sess/s)")
    print(f"Shim-eligible steps: {shim_eligible}")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_file}")


if __name__ == "__main__":
    raise SystemExit(main())
