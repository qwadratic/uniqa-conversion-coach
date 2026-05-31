"""
Prepare UNIFIED SFT from persona_v5: one model, all personas, short persona tag.

v5 has two SFT files:
  - static_sft.jsonl   (baseline persona behaviour, no coach)
  - coached_sft.jsonl  (persona behaviour WITH coach interventions)

Both already in {messages: [system, user, assistant]} format.
We strip the ~12K system prompt → short "persona: X" tag.

    python slurm/prepare_sft_v5_unified.py

Writes slurm/data_v5_unified/{train,val}.jsonl + summary.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


def detect_persona(system_content: str) -> str:
    m = re.search(r"Persona:\s*(\w+)", system_content)
    return m.group(1).lower() if m else "unknown"


def detect_step(user_content: str) -> str:
    m = re.search(r'"you_are_on":\s*"(\w+)"', user_content)
    return m.group(1) if m else "unknown"


def convert_row(row: dict, source: str) -> dict | None:
    msgs = row["messages"]
    if len(msgs) < 3:
        return None

    # Validate assistant output is valid JSON
    try:
        json.loads(msgs[2]["content"])
    except Exception:
        return None

    persona = detect_persona(msgs[0]["content"])
    step = detect_step(msgs[1]["content"])

    return {
        "messages": [
            {"role": "system", "content": f"persona: {persona}"},
            msgs[1],  # user (step context) — keep as-is
            msgs[2],  # assistant (JSON output) — keep as-is
        ],
        "_persona": persona,
        "_step": step,
        "_source": source,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datasets/persona_v5")
    ap.add_argument("--out", default="slurm/data_v5_unified")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    src = Path(args.dir)
    all_rows = []

    for fname, source in [("static_sft.jsonl", "static"), ("coached_sft.jsonl", "coached")]:
        fpath = src / fname
        if not fpath.exists():
            print(f"[warn] {fpath} not found, skipping")
            continue
        raw = [json.loads(l) for l in fpath.open()]
        converted = [r for r in (convert_row(r, source) for r in raw) if r is not None]
        print(f"  {fname}: {len(raw)} raw → {len(converted)} converted")
        all_rows.extend(converted)

    print(f"  total: {len(all_rows)} rows")

    # Stratified split by persona×step×source
    rng = random.Random(args.seed)
    groups: dict[str, list] = {}
    for r in all_rows:
        key = f"{r['_persona']}_{r['_step']}_{r['_source']}"
        groups.setdefault(key, []).append(r)

    train, val = [], []
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
                fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "total_train": len(train),
        "total_val": len(val),
        "sources": dict(Counter(r["_source"] for r in all_rows)),
        "by_persona": {},
    }
    for persona in ["judith", "franz", "peter"]:
        p_all = [r for r in all_rows if r["_persona"] == persona]
        p_train = [r for r in train if r["_persona"] == persona]
        p_val = [r for r in val if r["_persona"] == persona]
        summary["by_persona"][persona] = {
            "total": len(p_all),
            "train": len(p_train),
            "val": len(p_val),
            "steps": dict(Counter(r["_step"] for r in p_all)),
            "sources": dict(Counter(r["_source"] for r in p_all)),
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Size comparison
    sample_raw = json.loads(open(src / "static_sft.jsonl").readline())
    orig_len = len(sample_raw["messages"][0]["content"])
    new_len = len("persona: peter")
    print(f"\nSystem prompt: {orig_len} → {new_len} chars ({orig_len // new_len}x reduction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
