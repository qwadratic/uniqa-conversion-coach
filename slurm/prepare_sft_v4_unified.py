"""
Prepare UNIFIED SFT dataset from persona_v4: one model, all personas, short persona tag.

Strips the massive ~9K system prompt, replaces with a short persona tag.
The model learns persona-specific behaviour purely from the data patterns.

    python slurm/prepare_sft_v4_unified.py \
        --in datasets/persona_v4/sft_steps.jsonl \
        --out slurm/data_v4_unified

Writes:
  - train.jsonl  (all personas pooled, shuffled)
  - val.jsonl
  - summary.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def strip_to_tag(row: dict) -> dict | None:
    """Replace the giant system prompt with a short persona tag."""
    try:
        json.loads(row["output"])  # only keep valid JSON outputs
    except Exception:
        return None

    persona = row["persona"]
    step = row["step"]
    msgs = list(row["input_messages"])

    # Replace system message with short tag
    msgs[0] = {
        "role": "system",
        "content": f"persona: {persona}"
    }

    # Keep user message (step context) and add assistant output
    msgs.append({"role": "assistant", "content": row["output"]})

    return {
        "messages": msgs,
        "persona": persona,  # metadata for stratified split
        "step": step,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="datasets/persona_v4/sft_steps.jsonl")
    ap.add_argument("--out", default="slurm/data_v4_unified")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.inp).open()]
    converted = [r for r in (strip_to_tag(r) for r in rows) if r is not None]

    # Stratified split: val_frac per persona×step combo
    rng = random.Random(args.seed)
    rng.shuffle(converted)

    train, val = [], []
    # Group by persona+step for stratified split
    groups: dict[str, list] = {}
    for r in converted:
        key = f"{r['persona']}_{r['step']}"
        groups.setdefault(key, []).append(r)

    for key, items in groups.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * args.val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split, data in [("train", train), ("val", val)]:
        with (out / f"{split}.jsonl").open("w") as fh:
            for r in data:
                # Write only messages (drop metadata for training)
                fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "total_train": len(train),
        "total_val": len(val),
        "by_persona": {},
    }
    for persona in ["judith", "franz", "peter"]:
        p_train = [r for r in train if r["persona"] == persona]
        p_val = [r for r in val if r["persona"] == persona]
        summary["by_persona"][persona] = {
            "train": len(p_train),
            "val": len(p_val),
            "steps": dict(Counter(r["step"] for r in p_train + p_val)),
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Show size comparison
    sample = converted[0]
    orig_sys_len = len(rows[0]["input_messages"][0]["content"])
    new_sys_len = len(sample["messages"][0]["content"])
    print(f"\nSystem prompt: {orig_sys_len} chars → {new_sys_len} chars ({orig_sys_len/new_sys_len:.0f}x reduction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
