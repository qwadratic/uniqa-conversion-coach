"""
Parametrizable funnel dropoff evaluator — NO coach arm.

Measures φ = per-step churn + conversion + advisor rates for a persona mix,
then computes ε = mean absolute deviation vs UNIQA ground-truth anchors.

Works with two backends:
  teacher  — sim_loop LLM persona (needs OPENROUTER_API_KEY; runs locally)
  lora     — per-persona LoRA adapters (needs GPU; runs on Leonardo)

Usage — teacher (local, no GPU):
    python slurm/eval_dropoff.py \\
        --backend teacher --n 50 --proportions 0.30,0.50,0.20 \\
        --out eval_dropoff_teacher.json

Usage — LoRA (Leonardo):
    python slurm/eval_dropoff.py \\
        --backend lora \\
        --base ~/models/minicpm5-1b \\
        --adapter_dir ~/zero-one/leonardo/out_per_persona \\
        --n 100 --proportions 0.30,0.50,0.20 \\
        --out ~/zero-one/leonardo/dropoff_lora.json

Outputs JSON with:
  per_step_churn     : {S1..S8: leave_rate among sessions that reached this step}
  per_persona_conv   : {judith/franz/peter: {convert, abandon, advisor}}
  overall            : {convert_rate, abandon_rate, advisor_rate}
  epsilon            : TV distance from UNIQA anchors
  epsilon_breakdown  : per-step contribution
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "sim_loop"))

import widget as wdgt

# ── UNIQA ground-truth anchors ────────────────────────────────────────────────
# Source: funnel_autopsy + track brief. Churn = P(leave | reached step).
ANCHORS = {
    "S4_TARIFF_SELECT":   0.667,   # 66.7% leave at tariff/price step
    "S5_ADDON_SELECT":    0.240,   # 24% leave at add-on step
    "S6_PERSONAL_DATA":   0.390,   # ≈ combined S6+S7 ≈78% → split ~39/39
    "S7_HEALTH_QUESTIONS":0.390,
    "overall_convert":    0.056,   # 5.6% final conversion
}
PERSONA_MIX_DEFAULT = {"judith": 0.30, "franz": 0.50, "peter": 0.20}
PERSONA_SUCCESS = {
    "judith": {"convert", "advisor_handoff"},
    "franz":  {"convert"},
    "peter":  {"advisor_handoff", "convert"},
}
HESITATION_EVENTS = {
    "hover", "price_hover", "cancel_hover", "slow_mouse", "nav_back", "scroll_up",
    "idle", "pause", "tab_blur", "external_nav", "exit_intent", "validation_error",
    "field_clear", "tooltip_open", "rage_click", "text_select", "copy",
}

STATE_KEYS = ["attention", "satisfaction", "effort_left", "grasp", "effort_vs_reward"]
STEPS = wdgt.STEP_ORDER


# ── session instance sampler ──────────────────────────────────────────────────

_POOLS_PATH = _REPO / "sim_loop" / "session_pools.json"
_POOLS = json.loads(_POOLS_PATH.read_text()) if _POOLS_PATH.exists() else None


def _sample_instance(rng: random.Random) -> dict:
    if _POOLS is None:
        return {}
    p = _POOLS["pools"]
    return {k: rng.choice(v) for k, v in p.items()}


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


# ── outcome classifier ────────────────────────────────────────────────────────

def _classify_outcome(decision: str, reason: str, last_step: bool) -> str:
    if decision in ("continue", "advance") and last_step:
        return "convert"
    if decision == "leave":
        r = (reason or "").lower()
        if any(k in r for k in ("advisor", "berater", "telefon", "anruf",
                                "whatsapp", "callback", "rückruf", "kontakt")):
            return "advisor_handoff"
        return "abandon"
    return "in_progress"


# ── teacher backend ───────────────────────────────────────────────────────────

def _run_session_teacher(persona: str, rng: random.Random) -> dict:
    """Run one session through the funnel using the LLM teacher (sim_loop)."""
    from persona import LLMPersona

    si = _sample_instance(rng)
    start_state = dict(zip(STATE_KEYS, [1.0, 0.7, 1.0, 1.0, 0.7]))
    p = LLMPersona(persona, si, start_state)

    steps_detail = []
    outcome = "abandon"
    feed: list[dict] = []

    for step_idx, step_name in enumerate(STEPS):
        last = step_idx == len(STEPS) - 1
        screen = wdgt.render(step_name, dict(p.state), list(p.history_brief),
                             si, p.initial_intent,
                             coach_injection=None, recent_feed=feed[-12:])
        out = p.step(screen)
        events = out.get("events", []) if isinstance(out.get("events"), list) else []
        for e in events:
            if isinstance(e, dict):
                e.setdefault("step", step_name)
                feed.append(e)

        decision_raw = out.get("decision") or out.get("continuation") or "leave"
        decision = "continue" if decision_raw in ("continue", "advance", "more") else decision_raw
        reason = out.get("reason", "")

        steps_detail.append({"step": step_name, "decision": decision})
        result = _classify_outcome(decision, reason, last)
        if result != "in_progress":
            outcome = result
            break
        elif last:
            outcome = "convert"

    return {"persona": persona, "outcome": outcome,
            "steps": steps_detail, "n_steps": len(steps_detail)}


# ── LoRA backend ──────────────────────────────────────────────────────────────

class _LoRABackend:
    """Loads one LoRA adapter per persona and runs batched inference."""

    def __init__(self, base: str, adapter_dir: str, max_new_tokens: int = 512,
                 batch_size: int = 32):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._batch_size = batch_size
        self._max_new = max_new_tokens
        self._adapter_dir = Path(adapter_dir)

        self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        self._models: dict[str, object] = {}
        for persona in ("judith", "franz", "peter"):
            adp = self._adapter_dir / persona
            if not adp.exists():
                print(f"[warn] adapter not found: {adp}, skipping", flush=True)
                continue
            base_m = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True)
            self._models[persona] = PeftModel.from_pretrained(base_m, str(adp))
            self._models[persona].eval()
            print(f"[lora] loaded {persona} from {adp}", flush=True)

    def _infer(self, persona: str, msgs_list: list[list[dict]]) -> list[str]:
        model = self._models.get(persona)
        if model is None:
            return ['{"events":[],"decision":"leave","reason":"no_adapter"}'] * len(msgs_list)
        results = []
        for i in range(0, len(msgs_list), self._batch_size):
            chunk = msgs_list[i:i + self._batch_size]
            texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            plen = enc["input_ids"].shape[1]
            with self._torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=self._max_new,
                    do_sample=True, temperature=0.9, top_p=0.95,
                    pad_token_id=self.tok.pad_token_id)
            results.extend([self.tok.decode(g[plen:], skip_special_tokens=True) for g in gen])
        return results

    def run_cohort(self, persona: str, n: int, seed: int,
                   system_prompt: str | None = None) -> list[dict]:
        """Run N sessions for one persona through the funnel. Returns session dicts."""
        # Init sessions
        sessions = []
        for i in range(n):
            rng = random.Random(seed + i)
            si = _sample_instance(rng)
            sessions.append({
                "rng": rng, "si": si,
                "state": dict(zip(STATE_KEYS, [1.0, 0.7, 1.0, 1.0, 0.7])),
                "history_brief": [],
                "feed": [],
                "active": True,
                "outcome": "abandon",
                "n_steps": 0,
                "steps_detail": [],
            })

        sys_content = system_prompt or f"persona: {persona}"

        for step_idx, step_name in enumerate(STEPS):
            last = step_idx == len(STEPS) - 1
            active = [(i, s) for i, s in enumerate(sessions) if s["active"]]
            if not active:
                break

            # Build batch
            msgs_list = []
            for _, s in active:
                screen = wdgt.render(step_name, dict(s["state"]), list(s["history_brief"]),
                                     s["si"], s["si"].get("visit_goal", "researching"),
                                     coach_injection=None, recent_feed=s["feed"][-12:])
                msgs_list.append([
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": json.dumps(screen, ensure_ascii=False)},
                ])

            raws = self._infer(persona, msgs_list)

            for (_, s), raw in zip(active, raws):
                try:
                    out = json.loads(_strip_fences(raw))
                except Exception:
                    out = {"events": [], "decision": "leave", "reason": "parse_error"}

                events = out.get("events", []) if isinstance(out.get("events"), list) else []
                for e in events:
                    if isinstance(e, dict):
                        e.setdefault("step", step_name)
                        s["feed"].append(e)

                if isinstance(out.get("state"), dict):
                    for k in s["state"]:
                        try:
                            s["state"][k] = float(out["state"].get(k, s["state"][k]))
                        except (TypeError, ValueError):
                            pass

                decision_raw = out.get("decision") or out.get("continuation") or "leave"
                decision = ("continue" if decision_raw in ("continue", "advance", "more")
                            else decision_raw)
                reason = out.get("reason", "")
                s["history_brief"].append(f"{step_name}: {decision}/{out.get('feeling','')}")
                s["n_steps"] += 1
                s["steps_detail"].append({"step": step_name, "decision": decision})

                result = _classify_outcome(decision, reason, last)
                if result != "in_progress":
                    s["outcome"] = result
                    s["active"] = False
                elif last:
                    s["outcome"] = "convert"
                    s["active"] = False

        for s in sessions:
            if s["active"]:
                s["outcome"] = "convert"

        return [{"persona": persona, "outcome": s["outcome"],
                 "steps": s["steps_detail"], "n_steps": s["n_steps"]}
                for s in sessions]


# ── analysis ──────────────────────────────────────────────────────────────────

def _compute_report(all_sessions: list[dict]) -> dict:
    """Compute per-step churn, outcomes, and ε vs UNIQA anchors."""
    # Per-step: how many sessions reached this step, how many left
    step_entered: dict[str, int] = defaultdict(int)
    step_left: dict[str, int] = defaultdict(int)

    for s in all_sessions:
        for sd in s["steps"]:
            step_entered[sd["step"]] += 1
            if sd["decision"] == "leave":
                step_left[sd["step"]] += 1

    per_step_churn = {}
    for step in STEPS:
        n = step_entered.get(step, 0)
        l = step_left.get(step, 0)
        per_step_churn[step] = round(l / n, 3) if n > 0 else None

    # Per-persona outcomes
    by_persona: dict[str, list] = defaultdict(list)
    for s in all_sessions:
        by_persona[s["persona"]].append(s)

    per_persona = {}
    for persona, sess in by_persona.items():
        n = len(sess)
        outcomes = Counter(s["outcome"] for s in sess)
        success = sum(1 for s in sess if s["outcome"] in PERSONA_SUCCESS[persona])
        per_persona[persona] = {
            "n": n,
            "convert": round(outcomes.get("convert", 0) / n, 3),
            "abandon": round(outcomes.get("abandon", 0) / n, 3),
            "advisor": round(outcomes.get("advisor_handoff", 0) / n, 3),
            "success_rate": round(success / n, 3),
        }

    # Overall
    total = len(all_sessions)
    out_counter = Counter(s["outcome"] for s in all_sessions)
    overall = {
        "n": total,
        "convert_rate": round(out_counter.get("convert", 0) / total, 3),
        "abandon_rate": round(out_counter.get("abandon", 0) / total, 3),
        "advisor_rate": round(out_counter.get("advisor_handoff", 0) / total, 3),
    }

    # ε = mean |sim - anchor| over key steps
    eps_parts = {}
    for step, anchor in ANCHORS.items():
        if step == "overall_convert":
            sim_val = overall["convert_rate"]
        else:
            sim_val = per_step_churn.get(step)
        if sim_val is not None:
            eps_parts[step] = round(abs(sim_val - anchor), 3)

    epsilon = round(sum(eps_parts.values()) / len(eps_parts), 3) if eps_parts else None

    return {
        "per_step_churn": per_step_churn,
        "per_persona": per_persona,
        "overall": overall,
        "epsilon": epsilon,
        "epsilon_breakdown": eps_parts,
        "anchors": ANCHORS,
    }


def _print_report(r: dict, label: str = "") -> None:
    if label:
        print(f"\n{'='*60}")
        print(f"MODEL: {label}")
        print(f"{'='*60}")

    print("\nPer-step churn (leave rate among sessions reaching each step):")
    for step in STEPS:
        val = r["per_step_churn"].get(step)
        anchor = ANCHORS.get(step)
        if val is not None:
            anchor_str = f"  anchor={anchor:.3f}  Δ={val-anchor:+.3f}" if anchor else ""
            bar = "█" * int(val * 30)
            print(f"  {step:<30} {val:.3f}  {bar}{anchor_str}")

    print("\nPer-persona outcomes:")
    for persona, stats in r["per_persona"].items():
        print(f"  {persona:<8} convert={stats['convert']:.3f}  "
              f"abandon={stats['abandon']:.3f}  advisor={stats['advisor']:.3f}  "
              f"success={stats['success_rate']:.3f}")

    print(f"\nOverall: convert={r['overall']['convert_rate']:.3f}  "
          f"abandon={r['overall']['abandon_rate']:.3f}  "
          f"advisor={r['overall']['advisor_rate']:.3f}")

    if r["epsilon"] is not None:
        print(f"\nε = {r['epsilon']:.4f}  (mean |sim − anchor|, lower is better)")
        for k, v in r["epsilon_breakdown"].items():
            print(f"  {k}: {v:.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Funnel dropoff eval — no coach")
    ap.add_argument("--backend", choices=["teacher", "lora"], default="teacher")
    ap.add_argument("--n", type=int, default=50, help="Sessions per persona")
    ap.add_argument("--proportions", default="0.30,0.50,0.20",
                    help="judith,franz,peter mix (must sum to 1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval_dropoff.json")
    # LoRA-specific
    ap.add_argument("--base", default=None, help="Base model path (lora backend)")
    ap.add_argument("--adapter_dir", default=None,
                    help="Dir with judith/franz/peter sub-dirs (lora backend)")
    ap.add_argument("--system_prompt", default=None,
                    help="Override system prompt (lora backend). "
                         "Use {persona} for substitution. Default: 'persona: {persona}'")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    props = [float(x) for x in args.proportions.split(",")]
    if len(props) != 3:
        print("proportions must be 3 comma-separated floats")
        return 1
    persona_n = {
        "judith": round(args.n * props[0]),
        "franz":  round(args.n * props[1]),
        "peter":  round(args.n * props[2]),
    }

    t0 = time.time()
    all_sessions: list[dict] = []

    if args.backend == "teacher":
        import concurrent.futures
        print(f"Backend: teacher (LLM via OpenRouter)")
        print(f"Persona counts: {persona_n}")

        def run_one(args_tuple):
            persona, idx = args_tuple
            rng = random.Random(args.seed + idx)
            return _run_session_teacher(persona, rng)

        tasks = [(p, i) for p, n in persona_n.items() for i in range(n)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
            futures = {ex.submit(run_one, t): t for t in tasks}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                all_sessions.append(fut.result())
                done += 1
                if done % 10 == 0:
                    print(f"  {done}/{len(tasks)} sessions done", flush=True)

    else:  # lora
        if not args.base or not args.adapter_dir:
            print("--base and --adapter_dir required for lora backend")
            return 1
        print(f"Backend: lora  base={args.base}  adapters={args.adapter_dir}")
        print(f"Persona counts: {persona_n}")
        backend = _LoRABackend(args.base, args.adapter_dir,
                               max_new_tokens=args.max_new_tokens,
                               batch_size=args.batch_size)
        for persona, n in persona_n.items():
            print(f"\n[{persona}] running {n} sessions...", flush=True)
            sys_p = (args.system_prompt.replace("{persona}", persona)
                     if args.system_prompt else None)
            sess = backend.run_cohort(persona, n, seed=args.seed, system_prompt=sys_p)
            all_sessions.extend(sess)
            oc = Counter(s["outcome"] for s in sess)
            print(f"  {oc}", flush=True)

    dt = time.time() - t0
    report = _compute_report(all_sessions)
    report["meta"] = {
        "backend": args.backend,
        "n_per_persona": persona_n,
        "total": len(all_sessions),
        "wall_sec": round(dt, 1),
        "seed": args.seed,
        "adapter_dir": args.adapter_dir,
        "system_prompt_override": args.system_prompt,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _print_report(report, label=args.backend)
    print(f"\nWall: {dt:.1f}s  |  wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
