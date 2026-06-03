"""
Split any unified JSONL dataset into per-persona train/val files.

Works with:
  - data_v4_unified/  (K-sampled, 500/step/persona — PREFERRED)
  - data_v5_unified/  (imbalanced, judy starved)

Persona is detected from system message: "persona: X" or "_persona" metadata.

Usage:
    # Split v4 (recommended)
    python slurm/prepare_sft_per_persona.py \\
        --src slurm/data_v4_unified \\
        --out slurm/data_per_persona

    # From raw datasets/persona_v5 SFT files
    python slurm/prepare_sft_per_persona.py \\
        --src datasets/persona_v5 --src_format v5 \\
        --out slurm/data_per_persona_v5
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def detect_persona(row: dict) -> str:
    # from metadata field
    if "_persona" in row:
        return row["_persona"]
    # from messages[0].content = "persona: X"
    msgs = row.get("messages", [])
    if msgs:
        m = re.search(r"persona:\s*(\w+)", msgs[0].get("content", ""), re.I)
        if m:
            return m.group(1).lower()
    return "unknown"


def detect_step(row: dict) -> str:
    if "_step" in row:
        return row["_step"]
    msgs = row.get("messages", [])
    if len(msgs) >= 2:
        m = re.search(r'"you_are_on":\s*"(\w+)"', msgs[1].get("content", ""))
        if m:
            return m.group(1)
    return "unknown"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source directory with train.jsonl + val.jsonl")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--src_format", choices=["unified", "v5"], default="unified",
                    help="unified=data_v4_unified format; v5=persona_v5 raw SFT")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--personas", default="judith,franz,peter")
    args = ap.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    personas = [p.strip() for p in args.personas.split(",")]
    rng = random.Random(args.seed)

    if args.src_format == "unified":
        all_rows = load_jsonl(src / "train.jsonl") + load_jsonl(src / "val.jsonl")
    else:  # v5: static_sft + coached_sft
        all_rows = load_jsonl(src / "static_sft.jsonl") + load_jsonl(src / "coached_sft.jsonl")

    print(f"Loaded {len(all_rows)} total rows from {src}")

    # Group by persona × step
    by_persona: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    skipped = 0
    for row in all_rows:
        persona = detect_persona(row)
        if persona not in personas:
            skipped += 1
            continue
        step = detect_step(row)
        # Keep only the messages field for SFT
        clean = {"messages": row["messages"]} if "messages" in row else row
        by_persona[persona][step].append(clean)

    if skipped:
        print(f"  skipped {skipped} rows (unknown persona)")

    # Per-persona stratified split
    summary: dict[str, dict] = {}
    for persona in personas:
        steps_map = by_persona[persona]
        if not steps_map:
            print(f"  [warn] no rows for {persona}")
            continue

        all_p = []
        for step_rows in steps_map.values():
            all_p.extend(step_rows)

        # Stratified by step
        train, val = [], []
        for step, items in steps_map.items():
            rng.shuffle(items)
            n_val = max(1, int(len(items) * args.val_frac))
            val.extend(items[:n_val])
            train.extend(items[n_val:])

        rng.shuffle(train)
        rng.shuffle(val)

        (out / f"{persona}.train.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n")
        (out / f"{persona}.val.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n")

        summary[persona] = {
            "train": len(train),
            "val": len(val),
            "total": len(all_p),
            "steps": {s: len(rows) for s, rows in steps_map.items()},
        }
        print(f"  {persona}: {len(train)} train / {len(val)} val  "
              f"steps={dict(Counter(detect_step(r) for r in all_p))}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote per-persona splits to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
