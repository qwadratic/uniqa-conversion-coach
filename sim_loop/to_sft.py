"""Convert persona_v5 session JSONL files into SFT training pairs.

For each step in each session, reconstructs the exact screen the persona saw
(using widget.render + persona_prompt.build_system_prompt) and writes:

  {messages: [
    {role: "system",    content: <build_system_prompt(seg, session_instance)>},
    {role: "user",      content: <json.dumps(widget.render(...))>},
    {role: "assistant", content: <json.dumps(persona_output)>},
  ]}

Usage:
  python sim_loop/to_sft.py --in-dir datasets/persona_v5 --out-dir datasets/persona_v5
  python sim_loop/to_sft.py --in-dir /tmp/smoke --out-dir /tmp/smoke --dry-run
"""
from __future__ import annotations
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import widget as wdgt
from persona_prompt import build_system_prompt

# session_pools.json lives next to this file
_HERE = pathlib.Path(__file__).resolve().parent
_POOLS = json.loads((_HERE / "session_pools.json").read_text(encoding="utf-8"))
START_STATE: dict = _POOLS["start_state"]


def _history_line(step_name: str, persona_out: dict) -> str:
    """Reproduce the history_brief line that persona.py appends after each step."""
    dec = persona_out.get("decision", "")
    feel = persona_out.get("feeling", "")
    return f"{step_name}: {dec}/{feel}"


def session_to_sft_pairs(session: dict) -> list[dict]:
    """Yield one SFT dict per recorded step."""
    seg = session["persona"]
    si = session["session_instance"]
    initial_intent = si.get("visit_goal", "researching")
    system_prompt = build_system_prompt(seg, si)

    history_brief: list[str] = []
    running_state: dict = dict(START_STATE)

    pairs: list[dict] = []
    for rec in session.get("steps", []):
        step_name = rec["step"]
        coach_dec = rec.get("coach_decision") or {}
        persona_out = rec.get("persona_output") or {}

        # reconstruct coach_injection exactly as run.py does:
        #   coach_injection = coach_dec["command"] if coach_dec.get("_acted") else None
        coach_injection: dict | None = None
        if coach_dec.get("_acted"):
            coach_injection = coach_dec.get("command")

        # reproduce the screen the persona saw
        screen = wdgt.render(
            step_name,
            dict(running_state),
            list(history_brief),
            si,
            initial_intent,
            coach_injection,
        )

        pairs.append({
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": json.dumps(screen, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(persona_out, ensure_ascii=False)},
            ]
        })

        # advance state exactly as persona.py does
        new_state = persona_out.get("state") or {}
        for k in running_state:
            try:
                running_state[k] = float(new_state.get(k, running_state[k]))
            except Exception:
                pass
        history_brief.append(_history_line(step_name, persona_out))

    return pairs


def convert_file(src: pathlib.Path, dst: pathlib.Path, dry_run: bool = False) -> int:
    pairs_all: list[dict] = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            session = json.loads(line)
            pairs_all.extend(session_to_sft_pairs(session))
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            for p in pairs_all:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(pairs_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  default="datasets/persona_v5")
    ap.add_argument("--out-dir", default="datasets/persona_v5")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_dir  = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)

    pairs = {
        "off": (in_dir / "sessions_coach_off.jsonl",
                out_dir / "static_sft.jsonl"),
        "on":  (in_dir / "sessions_coach_on.jsonl",
                out_dir / "coached_sft.jsonl"),
    }

    for arm, (src, dst) in pairs.items():
        if not src.exists():
            print(f"[to_sft] SKIP {src} (not found)", file=sys.stderr)
            continue
        n = convert_file(src, dst, dry_run=args.dry_run)
        label = "(dry-run)" if args.dry_run else f"-> {dst}"
        print(f"[to_sft] arm={arm}  pairs={n}  {label}")


if __name__ == "__main__":
    main()
