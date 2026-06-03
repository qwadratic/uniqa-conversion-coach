"""
Step 2 of split pipeline (runs LOCALLY with internet):

Reads persona_traces.jsonl, runs the coach teacher (OpenRouter) over each
shim-eligible step + a sample of non-shim steps (for NO_ACTION training data),
produces coach SFT pairs.

For shim-eligible steps (hesitation + leave): coach observes, may inject a widget,
and we also re-prompt the persona model to get the post-intervention reaction
(if persona model is available locally — otherwise just record the coach decision).

Usage:
    python slurm/coach_from_traces.py \
        --traces leonardo/coach_data/persona_traces.jsonl \
        --out leonardo/coach_data/coach_sft \
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "sim_loop"))

from coach import CoachModel, COACH_SYSTEM


def filtered_event(e: dict) -> dict:
    return {k: e.get(k) for k in ("step", "type", "target", "value", "t", "source") if k in e}


def process_session(session: dict, coach_budget: int = 2,
                    coach_model: str | None = None) -> list[dict]:
    """Run coach teacher over one session's traces. Returns coach SFT pairs."""
    coach = CoachModel(mode="active", model=coach_model, budget=coach_budget,
                       temperature=0.4)
    pairs = []

    # Rebuild the feed incrementally as we walk through steps
    feed_so_far: list[dict] = []

    for step_rec in session.get("steps", []):
        step = step_rec["step"]
        events = step_rec.get("events", [])

        # Append this step's events to the running feed
        for e in events:
            if isinstance(e, dict):
                feed_so_far.append(e)

        # Skip detection-only steps
        if step in CoachModel.DETECTION_ONLY:
            continue

        # Coach observes
        coach_dec = coach.observe(feed_so_far, step=step)

        # Build the observation context (what coach saw — same format as training)
        context = {
            "current_step": step,
            "activity_log": coach._filter_feed(feed_so_far),
            "annoyance_budget_left": coach.budget - coach.used + (1 if coach_dec.get("_acted") else 0),
        }

        # Clean output for SFT
        output = {k: v for k, v in coach_dec.items() if not k.startswith("_")}

        pairs.append({
            "messages": [
                {"role": "system", "content": "coach"},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
            ],
            "_step": step,
            "_acted": coach_dec.get("_acted", False),
            "_shim_eligible": step_rec.get("shim_eligible", False),
        })

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="persona_traces.jsonl from Leonardo")
    ap.add_argument("--out", default="leonardo/coach_data/coach_sft")
    ap.add_argument("--coach_model", default=None, help="OpenRouter model for coach")
    ap.add_argument("--coach_budget", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    traces_path = Path(args.traces)
    sessions = [json.loads(l) for l in traces_path.open() if l.strip()]
    print(f"Loaded {len(sessions)} sessions from {traces_path}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_pairs: list[dict] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_session, s, args.coach_budget, args.coach_model): i
                for i, s in enumerate(sessions)}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                pairs = fut.result()
                all_pairs.extend(pairs)
            except Exception as e:
                errors += 1
                sys.stderr.write(f"[session {idx}] failed: {e}\n")
            done = len([f for f in futs if f.done()])
            if done % 10 == 0:
                print(f"  {done}/{len(sessions)} sessions processed ({time.time()-t0:.0f}s)",
                      flush=True)

    dt = time.time() - t0
    print(f"\nDone: {len(all_pairs)} pairs from {len(sessions)} sessions in {dt:.1f}s "
          f"({errors} errors)", flush=True)

    # Stats
    acted = sum(1 for p in all_pairs if p.get("_acted"))
    shim = sum(1 for p in all_pairs if p.get("_shim_eligible"))
    by_step = Counter(p["_step"] for p in all_pairs)

    print(f"  acted: {acted}, no_action: {len(all_pairs) - acted}, shim-eligible: {shim}")
    print(f"  by step: {dict(by_step)}")

    # Split train/val (90/10, stratified by acted/not)
    rng = random.Random(args.seed)
    acted_pairs = [p for p in all_pairs if p.get("_acted")]
    noact_pairs = [p for p in all_pairs if not p.get("_acted")]

    rng.shuffle(acted_pairs)
    rng.shuffle(noact_pairs)

    def split_90_10(lst):
        n_val = max(1, int(len(lst) * 0.1))
        return lst[n_val:], lst[:n_val]

    train_a, val_a = split_90_10(acted_pairs)
    train_n, val_n = split_90_10(noact_pairs)
    train = train_a + train_n
    val = val_a + val_n
    rng.shuffle(train)
    rng.shuffle(val)

    # Write (strip internal fields)
    for split, data in [("train", train), ("val", val)]:
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for p in data:
                f.write(json.dumps({"messages": p["messages"]}, ensure_ascii=False) + "\n")

    summary = {
        "n_sessions": len(sessions),
        "total_pairs": len(all_pairs),
        "acted": acted,
        "no_action": len(all_pairs) - acted,
        "shim_eligible": shim,
        "train": len(train),
        "val": len(val),
        "by_step": dict(by_step),
        "wall_sec": round(dt, 1),
        "errors": errors,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_dir}/train.jsonl ({len(train)}), val.jsonl ({len(val)})")


if __name__ == "__main__":
    raise SystemExit(main())
