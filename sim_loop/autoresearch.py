"""Autoresearch (Loop B) — self-improve the COACH SYSTEM PROMPT via LLM trace analysis.

Core loop
─────────
  PROPOSE  a targeted edit to the coach system prompt, reading the last round's traces
           (which effector fired on which step → outcome + persona's intervention_assessment)
           and proposing ONE concrete, auditable prompt constraint via LLM.
  SIMULATE the on-arm with the edited prompt over a FIXED paired persona assignment.
  EVALUATE Δ_success = success_rate(on) − success_rate(off)
           using run.py's persona-dependent SUCCESS map (the right outcome differs by segment).
           Also track online-convert and annoyance (interventions/session).
  GATE     accept iff  success_uplift > incumbent_uplift + τ
                   AND annoyance ≤ ceiling
  REPEAT   accepted variant becomes the incumbent; its traces feed the next proposer call.

Key improvements over the prior directive-toggle loop
───────────────────────────────────────────────────────
- Mutation operator edits the ACTUAL prompt text (not a canned directive list).
- LLM proposer reads last round's traces: per-effector×step fire counts, per-persona
  success rates, and budget-exhaustion fraction → proposes ONE targeted constraint.
- Gate measures persona-SUCCESS uplift (not raw convert_rate), matching the formalization.
- Ledger records {round, prompt_diff, conv_off/on, success_off/on, uplift, annoyance, accepted}.
- Outputs: best_prompt.txt (the winning full system prompt) + report.md.

Known failure modes addressed (persona_v5 data, n=120):
  1. form_simplify over-fires on S4_TARIFF_SELECT (comparison screen, not a form): 48% of all acts.
  2. Budget exhausted by S4–S5 in 48% of sessions → nothing left for S7/S8 price walls.
  3. Franz (online-affine) hurt most: success 28%→22% (−6pp) from early interruptions.

Usage
─────
  # smoke test (fast, cheap):
  python sim_loop/autoresearch.py --rounds 2 --sessions 6 --concurrency 8

  # real run:
  python sim_loop/autoresearch.py --rounds 8 --sessions 20 --tau 0.03 --concurrency 8

  # fallback proposer (no LLM for mutation, uses curated list):
  python sim_loop/autoresearch.py --rounds 6 --sessions 12 --proposer mutate
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run as R                            # run_session, is_success, REAL, BALANCED
from coach import COACH_SYSTEM, EFFECTORS  # baseline prompt + full effector id list
from llm import chat, extract_json


# ── Targeted fallbacks addressing the persona_v5 failure modes ───────────────────
# Each is a concrete, falsifiable rule. The proposer appends them ONE at a time when
# the LLM proposer fails, working through the list in order of estimated impact.
TARGETED_FALLBACKS: list[str] = [
    (
        "NEVER use form_simplify outside S3_PERSONAL_INFO and S6_PERSONAL_DATA. "
        "S4_TARIFF_SELECT is a price/comparison screen, not a form; firing form_simplify "
        "there signals irrelevance and wastes budget."
    ),
    (
        "Reserve budget: spend at most 1 intervention before S6_PERSONAL_DATA. "
        "Hold the remaining budget for S7_HEALTH_QUESTIONS and S8_REVIEW_PURCHASE — "
        "the two steps with the highest dropout rates and where help matters most."
    ),
    (
        "For fast/decisive users (Franz signal: linear progression through S1→S3 with "
        "no back-navigation, no tooltip opens, and dwell < 30 s per step): "
        "stay completely silent through S1–S4 and save ALL budget for S7/S8."
    ),
    (
        "At S4_TARIFF_SELECT the only appropriate effectors are: price_reframe, "
        "pricing_explain, package_nuance, preselect_optimal, upgrade_explain, "
        "feature_highlight, or quick_quiz. "
        "NEVER form_simplify, form_explainer, form_helper, or field_defer at S4."
    ),
    (
        "At S5_ADDON_SELECT: prefer addon_skip_ok (87% engagement) as the default "
        "first-and-only intervention. Only escalate to a different effector if "
        "addon_skip_ok was already used this session."
    ),
    (
        "For Franz (high-confidence: fast + no hesitation signals): NEVER offer "
        "advisor_handoff, callback_offer, whatsapp_bot, or contact_handoff — "
        "those are FAILURE outcomes for Franz and increase his abandon rate."
    ),
    (
        "At S7_HEALTH_QUESTIONS: if the price likely increased after health questions "
        "(prolonged dwell after the price reveal, or re-scroll), use health_explain or "
        "value_justification BEFORE the user reaches the exit-intent threshold."
    ),
]


# ── Policy dataclass ──────────────────────────────────────────────────────────────
@dataclass
class PromptPolicy:
    """A coach policy: full system prompt text + hyper-params + audit trail."""

    prompt_text: str        # the full COACH_SYSTEM text (may have appended constraints)
    budget: int = 2         # annoyance budget passed to CoachModel
    temperature: float = 0.4
    diff: str = "baseline"  # one-line audit: what changed from the baseline

    @property
    def system(self) -> str:
        """Passed as coach_system= to run_session."""
        return self.prompt_text

    @property
    def label(self) -> str:
        d = self.diff[:80] + "…" if len(self.diff) > 80 else self.diff
        return f"budget={self.budget} temp={self.temperature} | {d!r}"

    def to_dict(self) -> dict:
        return {
            "budget": self.budget,
            "temperature": self.temperature,
            "diff": self.diff,
            "prompt_length": len(self.prompt_text),
        }


# Baseline: the original prompt, budget=2 (persona_v5 used 3 → exhausted; 2 is safer)
BASE = PromptPolicy(prompt_text=COACH_SYSTEM, budget=2, temperature=0.4, diff="baseline")


# ── Trace analysis ────────────────────────────────────────────────────────────────
def summarize_traces(results: list[dict]) -> dict:
    """Extract actionable signals: which effectors fired on which steps + outcomes."""
    persona_data: dict = defaultdict(lambda: {
        "total": 0, "success": 0, "abandon": 0, "interventions": 0
    })
    fire_map: dict = defaultdict(lambda: {"count": 0, "success_count": 0})

    for r in results:
        p = r["persona"]
        persona_data[p]["total"] += 1
        if R.is_success(p, r["outcome"]):
            persona_data[p]["success"] += 1
        if r["outcome"] == "abandon":
            persona_data[p]["abandon"] += 1
        persona_data[p]["interventions"] += r.get("coach_interventions", 0)

        for step_rec in r.get("steps", []):
            cd = step_rec.get("coach_decision") or {}
            if cd.get("_acted"):
                eff = (cd.get("command") or {}).get("effector", "NO_ACTION")
                step = step_rec.get("step", "?")
                key = f"{eff}@{step}"
                fire_map[key]["count"] += 1
                if R.is_success(p, r["outcome"]):
                    fire_map[key]["success_count"] += 1

    top = sorted(fire_map.items(), key=lambda x: -x[1]["count"])[:18]
    low_success = [
        {
            "effector_at_step": k,
            "fires": v["count"],
            "successes": v["success_count"],
            "success_rate": round(v["success_count"] / v["count"], 2),
        }
        for k, v in top
        if v["count"] >= 2 and v["success_count"] / v["count"] < 0.35
    ][:6]

    total = len(results) or 1
    n_exhausted = sum(
        1 for r in results
        if r.get("coach_interventions", 0) >= BASE.budget
    )

    return {
        "session_count": total,
        "budget_exhausted_fraction": round(n_exhausted / total, 2),
        "per_persona": {
            p: {
                "total": v["total"],
                "success_rate": round(v["success"] / v["total"], 2) if v["total"] else 0,
                "abandon_rate": round(v["abandon"] / v["total"], 2) if v["total"] else 0,
                "avg_interventions": round(v["interventions"] / v["total"], 2) if v["total"] else 0,
            }
            for p, v in persona_data.items()
        },
        "top_effector_step_fires": [
            {
                "key": k,
                "fires": v["count"],
                "successes": v["success_count"],
                "success_rate": round(v["success_count"] / v["count"], 2) if v["count"] else 0,
            }
            for k, v in top[:12]
        ],
        "low_success_fire_patterns": low_success,
    }


# ── Prompt edit helpers ──────────────────────────────────────────────────────────
_CONSTRAINT_MARKER = "\n\n## AUTORESEARCH CONSTRAINTS (applied, do not override):\n"


def _apply_constraint(prompt_text: str, directive: str) -> str:
    """Append one targeted constraint to the prompt, under a deduplicated section header."""
    if directive in prompt_text:
        return prompt_text          # already present — idempotent
    if _CONSTRAINT_MARKER in prompt_text:
        return prompt_text.rstrip() + f"\n- {directive}\n"
    return prompt_text + _CONSTRAINT_MARKER + f"- {directive}\n"


# ── Proposers ────────────────────────────────────────────────────────────────────
def propose_prompt_edit(
    policy: PromptPolicy,
    on_results: list[dict],
    model: str | None,
    rng: random.Random,
) -> PromptPolicy:
    """LLM-first proposer: reads last round's traces, proposes ONE targeted constraint.
    Falls back to curated TARGETED_FALLBACKS on any LLM failure."""
    summary = summarize_traces(on_results)

    # Extract any constraints already in the prompt to avoid re-proposing them
    existing: list[str] = []
    if _CONSTRAINT_MARKER in policy.prompt_text:
        section = policy.prompt_text.split(_CONSTRAINT_MARKER, 1)[1]
        existing = [
            ln.lstrip("- ").strip()
            for ln in section.splitlines()
            if ln.strip().startswith("-")
        ]

    sys_msg = (
        "You are an expert at improving insurance-funnel conversion-coach system prompts.\n"
        "You receive a trace summary showing which coach effectors fired on which funnel "
        "steps, their per-session success rates, and per-persona outcomes.\n"
        "Propose ONE short, concrete, TARGETED constraint to append to the coach system "
        "prompt that addresses the most damaging failure pattern visible in the traces.\n\n"
        "Good constraint examples:\n"
        "  - 'NEVER use form_simplify outside S3_PERSONAL_INFO and S6_PERSONAL_DATA'\n"
        "  - 'Hold at least 1 budget unit for S7 or later'\n"
        "  - 'For Franz (fast, linear, no hesitation): stay silent through S1–S4'\n\n"
        "The constraint must be:\n"
        "  1. Specific: names the effector(s), step(s), or persona(s) it constrains\n"
        "  2. Actionable: tells the coach what TO DO or NOT DO\n"
        "  3. Different from the existing constraints listed in the request\n\n"
        "Reply ONLY as JSON:\n"
        '{"directive": "<one concrete rule sentence>", '
        '"addresses": "<which trace pattern this fixes>", '
        '"persona_scope": "<franz|judith|peter|all>"}'
    )

    user_msg = json.dumps(
        {
            "trace_summary": summary,
            "constraints_already_in_prompt": existing,
            "known_baseline_failures": [
                "form_simplify over-fires on S4_TARIFF_SELECT (price/comparison screen) — 48% of all acts in prior run",
                "Budget exhausted before S7/S8 in 48% of sessions — price walls missed",
                "Franz (fast, online-affine) success rate dropped −6pp from early interruptions",
            ],
        },
        ensure_ascii=False,
    )

    try:
        raw = chat(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": user_msg}],
            model=model, temperature=0.7, max_tokens=250,
        )
        d = extract_json(raw)
        directive = (d.get("directive") or "").strip()
        if directive and len(directive) > 15 and directive not in policy.prompt_text:
            new_text = _apply_constraint(policy.prompt_text, directive)
            return PromptPolicy(
                new_text, policy.budget, policy.temperature,
                diff=f"LLM→ {directive[:100]}"
            )
    except Exception as e:
        sys.stderr.write(f"[propose] LLM proposer failed ({type(e).__name__}), using fallback\n")

    return _fallback_mutate(policy, rng)


def _fallback_mutate(policy: PromptPolicy, rng: random.Random) -> PromptPolicy:
    """Pick the first unused directive from TARGETED_FALLBACKS; nudge budget as last resort."""
    for directive in TARGETED_FALLBACKS:
        # Use first ~50 chars as a dedup key
        if directive[:50] not in policy.prompt_text:
            new_text = _apply_constraint(policy.prompt_text, directive)
            return PromptPolicy(
                new_text, policy.budget, policy.temperature,
                diff=f"fallback→ {directive[:80]}"
            )
    # All fallbacks already applied — nudge budget ±1 within [1, 3]
    new_b = max(1, min(3, policy.budget + rng.choice([-1, 1])))
    return PromptPolicy(
        policy.prompt_text, new_b, policy.temperature,
        diff=f"budget {policy.budget}→{new_b}"
    )


# ── Evaluation ───────────────────────────────────────────────────────────────────
def run_arm(
    assign: list[str],
    arm: str,
    policy: PromptPolicy,
    model: str | None,
    concurrency: int,
    base_seed: int,
) -> list[dict]:
    """Run one arm (off or on) over the pre-sampled persona assignment."""
    jobs: list = []
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, seg in enumerate(assign):
            jobs.append(ex.submit(
                R.run_session,
                seg, arm, model, policy.budget,
                base_seed * 100000 + i,
                coach_system=policy.system,
                coach_temperature=policy.temperature,
            ))
        for fut in as_completed(jobs):
            try:
                results.append(fut.result())
            except Exception as e:
                sys.stderr.write(f"[session] {e}\n")
    return results


def metrics(results: list[dict]) -> dict:
    """Compute conversion, persona-success, advisor-handoff, abandon, and annoyance."""
    n = len(results) or 1
    conv = sum(1 for r in results if r["outcome"] == "convert")
    adv = sum(1 for r in results if r["outcome"] == "advisor_handoff")
    ab = sum(1 for r in results if r["outcome"] == "abandon")
    succ = sum(1 for r in results if R.is_success(r["persona"], r["outcome"]))
    interv = sum(r["coach_interventions"] for r in results)
    return {
        "n": len(results),
        "convert": conv,
        "convert_rate": round(conv / n, 3),
        "advisor": adv,
        "abandon": ab,
        "success": succ,
        "success_rate": round(succ / n, 3),
        "interventions_per_session": round(interv / n, 2),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Coach prompt autoresearch loop")
    ap.add_argument("--rounds", type=int, default=6,
                    help="number of propose→evaluate→gate iterations")
    ap.add_argument("--sessions", type=int, default=12,
                    help="sessions per arm (same assignment, paired)")
    ap.add_argument("--tau", type=float, default=0.02,
                    help="acceptance margin: success_uplift must exceed incumbent by τ")
    ap.add_argument("--annoyance-ceiling", type=float, default=2.0,
                    help="max interventions/session on the ON arm (annoyance gate)")
    ap.add_argument("--proposer", choices=["llm", "mutate"], default="llm",
                    help="llm=LLM trace-reader (default); mutate=targeted-fallback list only")
    ap.add_argument("--proportions", choices=["real", "balanced"], default="real")
    ap.add_argument("--model", default=None,
                    help="OpenRouter model override (default: OPENROUTER_MODEL env var)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sim_loop/out/autoresearch")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    weights = R.REAL if args.proportions == "real" else R.BALANCED
    pool = list(weights.keys())
    wts = [weights[p] for p in pool]
    assign = [rng.choices(pool, wts)[0] for _ in range(args.sessions)]

    # Cost estimate: off arm + base on arm + N candidate on arms, each × sessions × ~16 steps
    n_arms = 2 + args.rounds          # off, base-on, + one per round
    est_calls = n_arms * args.sessions * 16
    est_min = est_calls / args.concurrency * 2.5 / 60
    print(
        f"≈ {est_calls} LLM calls "
        f"({n_arms} arms × {args.sessions} sessions × ~16 steps) "
        f"@ concurrency {args.concurrency} → ~{est_min:.0f} min\n"
        f"Persona mix (pre-sampled): "
        + ", ".join(f"{p}={assign.count(p)}" for p in pool)
    )

    outd = pathlib.Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    ledger_path = outd / "ledger.jsonl"
    ledger_f = ledger_path.open("w", encoding="utf-8")

    def log(rec: dict) -> None:
        ledger_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ledger_f.flush()

    t0 = time.time()

    # ── Step 1: shared control (coach OFF) — run exactly once ─────────────────────
    print("\n[…] Running control arm (coach OFF)…")
    off_res = run_arm(assign, "off", BASE, args.model, args.concurrency, args.seed)
    m_off = metrics(off_res)
    conv_off = m_off["convert_rate"]
    success_off = m_off["success_rate"]
    print(
        f"[control  OFF] convert={conv_off}  success={success_off}  "
        f"(n={m_off['n']}  judith={assign.count('judith')} "
        f"franz={assign.count('franz')} peter={assign.count('peter')})"
    )
    log({"round": 0, "kind": "control_off", **m_off})

    # ── Step 2: incumbent = base prompt on-arm (run once as starting point) ────────
    print("[…] Running incumbent base on-arm…")
    on_res = run_arm(assign, "on", BASE, args.model, args.concurrency, args.seed)
    m_on = metrics(on_res)
    inc_success_uplift = round(m_on["success_rate"] - success_off, 3)
    incumbent: PromptPolicy = BASE
    inc_metrics = m_on
    inc_results = on_res
    print(
        f"[incumbent  ON] convert={m_on['convert_rate']}  success={m_on['success_rate']}  "
        f"uplift={inc_success_uplift:+.3f}  annoy={m_on['interventions_per_session']}"
    )
    log({
        "round": 0, "kind": "incumbent_base",
        "policy": incumbent.to_dict(),
        "prompt_diff": incumbent.diff,
        "conv_off": conv_off, "success_off": success_off,
        "conv_on": m_on["convert_rate"], "success_on": m_on["success_rate"],
        "uplift": inc_success_uplift,
        "annoyance": m_on["interventions_per_session"],
        "accepted": True,
    })

    # ── Step 3: propose → evaluate → gate ─────────────────────────────────────────
    for rnd in range(1, args.rounds + 1):
        print(f"[…] Round {rnd}/{args.rounds}: proposing edit…")
        if args.proposer == "llm":
            cand = propose_prompt_edit(incumbent, inc_results, args.model, rng)
        else:
            cand = _fallback_mutate(incumbent, rng)

        print(f"      diff: {cand.diff[:100]}")
        res = run_arm(assign, "on", cand, args.model, args.concurrency, args.seed)
        m = metrics(res)
        success_uplift = round(m["success_rate"] - success_off, 3)
        annoy = m["interventions_per_session"]
        accept = (
            success_uplift > inc_success_uplift + args.tau
            and annoy <= args.annoyance_ceiling
        )
        flag = "✅ ACCEPT" if accept else "·  reject"
        print(
            f"[round {rnd:2d}] {flag}  "
            f"success_on={m['success_rate']}  uplift={success_uplift:+.3f}  "
            f"(need >{inc_success_uplift + args.tau:.3f})  "
            f"annoy={annoy:.2f} (ceil {args.annoyance_ceiling})"
        )
        log({
            "round": rnd,
            "kind": "candidate",
            "policy": cand.to_dict(),
            "prompt_diff": cand.diff,
            "conv_off": conv_off,
            "success_off": success_off,
            "conv_on": m["convert_rate"],
            "success_on": m["success_rate"],
            "uplift": success_uplift,
            "incumbent_uplift": inc_success_uplift,
            "annoyance": annoy,
            "tau": args.tau,
            "annoyance_ceiling": args.annoyance_ceiling,
            "accepted": accept,
        })
        if accept:
            incumbent = cand
            inc_success_uplift = success_uplift
            inc_metrics = m
            inc_results = res
            print(f"           → new incumbent  (success_uplift={inc_success_uplift:+.3f})")

    ledger_f.close()

    # ── Outputs ────────────────────────────────────────────────────────────────────
    (outd / "best_prompt.txt").write_text(incumbent.prompt_text, encoding="utf-8")
    (outd / "best_policy.json").write_text(
        json.dumps(
            {
                "policy": incumbent.to_dict(),
                "conv_off": conv_off,
                "success_off": success_off,
                "success_uplift": inc_success_uplift,
                "metrics": inc_metrics,
                "rounds": args.rounds,
                "sessions_per_arm": args.sessions,
                "tau": args.tau,
                "annoyance_ceiling": args.annoyance_ceiling,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    wall = time.time() - t0
    tail = incumbent.prompt_text[-300:]
    report = (
        f"# Coach Autoresearch — Result\n\n"
        f"| Param | Value |\n"
        f"|---|---|\n"
        f"| rounds | {args.rounds} |\n"
        f"| sessions/arm | {args.sessions} |\n"
        f"| proposer | {args.proposer} |\n"
        f"| τ (accept threshold) | {args.tau} |\n"
        f"| annoyance ceiling | {args.annoyance_ceiling} |\n"
        f"| wall time | {wall:.0f}s |\n\n"
        f"## Headline numbers\n\n"
        f"| Metric | coach-OFF | best coach-ON | Δ |\n"
        f"|---|---|---|---|\n"
        f"| Online convert rate | {conv_off:.3f} | {inc_metrics['convert_rate']:.3f} | "
        f"{inc_metrics['convert_rate'] - conv_off:+.3f} |\n"
        f"| Persona-success rate | {success_off:.3f} | {inc_metrics['success_rate']:.3f} | "
        f"{inc_success_uplift:+.3f} |\n"
        f"| Interventions/session | — | {inc_metrics['interventions_per_session']:.2f} | — |\n\n"
        f"## Best prompt diff\n\n```\n{incumbent.diff}\n```\n\n"
        f"## Best prompt (tail)\n\n```\n…{tail}\n```\n\n"
        f"Ledger: `{ledger_path}`  \n"
        f"Best prompt: `{outd / 'best_prompt.txt'}`  \n"
        f"Best policy: `{outd / 'best_policy.json'}`\n"
    )
    (outd / "report.md").write_text(report, encoding="utf-8")

    print(
        f"\n{'='*60}\n"
        f"BEST  success_uplift={inc_success_uplift:+.3f}  over baseline {success_off}  "
        f"diff: {incumbent.diff[:80]}\n"
        f"Artifacts → {outd}\n"
        f"  ledger.jsonl · best_prompt.txt · best_policy.json · report.md"
    )


if __name__ == "__main__":
    main()
