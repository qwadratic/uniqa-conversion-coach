# ROADMAP — FINAL (Sat 23:15 → Sun 10:00, ≈10h45)

> **This is the execution plan we follow from now to submission. No re-planning.**

---

## 1. Situation read + the A-vs-B decision

We have a working detection+decision system end-to-end on synthetic data: 11-step funnel,
12-action coach with hard gates, JSON contracts, turn-based simulator, A/B harness,
empirical autoresearch gate, LLM persona engine (ε≈0.10 at N=300, ε=0.128 at N=500 regen
just under/over the 0.12 gate), React funnel twin + coach overlay layer with 5 demo
decisions wired, 76+ tests green. **The big missing thing is the live wow-moment: a coach
that visibly reads signals from the running webapp and intervenes — not just button-fired
canned JSON.**

**Decision: HYBRID, primary = B (Dynamic Coach), tail-risk = A (Leonardo persona LoRA in background).**

- **Primary (B, the shipped demo):** spend the awake hours wiring a **live LLM coach
  bridge** (OpenRouter `gpt-4o-mini`, ~300ms/decision) between the React webapp and a
  tiny Python WebSocket server, so judges see:
  (1) a human (or scripted persona) navigate the FunnelTwin →
  (2) activity events stream to the coach →
  (3) the coach returns reasoned, differentiated decisions →
  (4) the existing `CoachLayer` renders them →
  (5) a side panel shows per-persona uplift from the existing Monte-Carlo A/B.
- **Tail-risk (A, the HPC bullet):** kick off the existing Leonardo per-persona LoRA
  scaffold **in the background tonight**. If it lands by 09:00 → one extra bullet
  ("3 personas distilled to 1B local on CINECA Leonardo, ε≈X"). If it doesn't → cut it
  silently, the demo is unaffected.
- **Why not A as primary:** persona distillation is a *latency/cost* optimization for the
  autoresearch inner loop — it is **invisible to a judge** in 60 seconds. The plan §5.1
  is explicit that the *coach* (not the persona) is the real Leonardo subject, and coach
  distillation needs a trace dataset we haven't generated → not feasible in 10h.
- **Why hybrid not pure B:** Leonardo persona run is genuinely "fire and forget" — the
  scaffold exists (`slurm/{prepare_sft,train_persona_lora,slurm_finetune}.py`), the
  dataset exists (`datasets/persona_v1/sft_steps.jsonl`, 2122 pairs), the reservation
  window ends Sun 12:00 (after our deadline). Cost to launch ≈ 30 min; upside = real HPC bullet.

### Explicitly CUT / DEFERRED tonight
- ❌ **Coach distillation on Leonardo** (no trace dataset; out of time).
- ❌ **Z3 formal certificate** (already in `deferred/`, stay deferred).
- ❌ **Loop B / off-policy IPS eval** (no real data anyway).
- ❌ **New surfaces beyond what's already wired** (email, WA, save_progress already have demo decisions — don't add survey/LP/booking now).
- ❌ **Re-tuning persona dials** (regen at N=500 ε=0.128 is fine; either ship it or fall back to persona_v1 N=300 ε=0.10 — both are demonstrable).
- ❌ **Adding tests for new code beyond smoke-level** (carry the 76-test green flag; new bridge code gets one smoke test, no more).
- ❌ **Karpathy/Infineon and Sybilion side tracks** (not in scope tonight).

---

## 2. Time-boxed tasks (dependency order)

Time budget assumes **~5h of sleep around 03:30–08:00**, so all heavy lifting must clear by 03:30.

| # | When | Owner | Time | Task |
|---|------|-------|------|------|
| T0 | 23:15–23:30 | LOCAL | 15m | **Stop the regen, lock the dataset** |
| T1 | 23:30–00:15 | BG / LEO | 45m | **Kick off Leonardo persona LoRA (fire-and-forget)** |
| T2 | 00:15–01:45 | LOCAL | 90m | **Python coach bridge (FastAPI + OpenRouter)** |
| T3 | 01:45–02:45 | LOCAL | 60m | **Wire React webapp ↔ bridge (replace mock stream)** |
| T4 | 02:45–03:30 | DEMO | 45m | **Live A/B uplift sidebar + persona-autorun mode** |
| T5 | 03:30–08:00 | 💤 | 4.5h | **Sleep / buffer** |
| T6 | 08:00–08:30 | LEO | 30m | **Check Leonardo result; harvest or cut** |
| T7 | 08:30–09:15 | DEMO | 45m | **Final A/B numbers + README/judge story** |
| T8 | 09:15–09:45 | DEMO | 30m | **Screencast (fallback if live demo flakes)** |
| T9 | 09:45–10:00 | DEMO | 15m | **Submit** |

---

### T0 — Lock the persona dataset (23:15–23:30, LOCAL, 15m)

- Check `datasets/persona_v1_regen/sessions.jsonl` line count. Decide:
  - If regen is **complete (500 sessions)** and ε ≤ 0.12 on quick eval → promote it: `cp datasets/persona_v1_regen/* datasets/persona_v1/` (after backup).
  - Otherwise → **kill the regen** and **keep `datasets/persona_v1/` as-is** (the N=500 ε=0.128 / N=300 ε=0.10 lock from the manifest). Don't burn time chasing ε.
- Files touched: `datasets/persona_v1/` (decide which artifact ships).
- Acceptance: `datasets/persona_v1/sft_steps.jsonl` exists, `manifest.json` present, `report.md` quotes a number we can put on a slide.

---

### T1 — Kick off Leonardo persona LoRA (23:30–00:15, BG/LEO, 45m)

This is the only Leonardo work. **Fire-and-forget**, no babysitting after launch.

```bash
# 1. prepare SFT shards LOCALLY (login node has the dataset anyway, but do it once here too)
python slurm/prepare_sft.py --in datasets/persona_v1/sft_steps.jsonl --out slurm/data
# expect slurm/data/{judith,franz,peter}.{train,val}.jsonl + summary.json

# 2. connect to Leonardo (skill: leonardo-connect)
#    On LOGIN node, pre-stage base weights (compute nodes have NO internet):
ssh leonardo
cd $HOME/zero-one && git pull
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $HOME/models/qwen2.5-1.5b

# 3. rsync the prepared data + slurm script
$LEO put leonardo

# 4. submit the job — partition boost_usr_prod, reservation s_tra_ncc (ends Sun 12:00)
$LEO run "cd zero-one && sbatch slurm/slurm_finetune.sh"
# note the JOBID; the job runs ~1.5h sequentially across the 3 personas + eval
```

- Files touched: `slurm/slurm_finetune.sh` (may need to override `BASE` to the
  pre-staged path: `export BASE=$HOME/models/qwen2.5-1.5b` in the sbatch env).
- Acceptance: `squeue --me` shows the job RUNNING or PENDING with a reasonable ETA.
  Move on — **do not wait for it.** If anything in this task takes >45 min, abort
  cleanly (`scancel`), document "Leonardo deferred — scaffold ready, didn't ship",
  and proceed. The demo does not depend on this.

---

### T2 — Python coach bridge (00:15–01:45, LOCAL, 90m)

The single most demo-critical piece. Build a small WebSocket bridge that wraps the
existing rule coach + an OpenRouter LLM coach behind one endpoint the webapp can hit.

**New file: `sim_loop/coach.py`** (≈120 LOC)

```python
# FastAPI WS server. Listens for activity events from the React Recorder,
# maintains a session-scoped CoachObservation, calls a CoachModel.decide(obs),
# pushes resulting CoachDecision JSON back over the same socket.
#
# Two coach backends, toggle via query param ?backend=rule|llm :
#   - "rule": existing sim_loop/coach.py decide_action() -> map via coach_io
#   - "llm":  OpenRouter gpt-4o-mini with a system prompt embedding
#             coach.py's hard constraints + COACH_MODIFIERS table (reuse persona_datagen's
#             OpenRouter call helper for plumbing)
#
# Throttle: at most 1 decision per ~2s per session, and respect MESSAGE_BUDGET=3.
# Output JSON must match sim_loop/replay/src/coachWidgets.tsx CoachDecision shape
# (effector + step + target? + payload + render? + reasoning + confidence).
```

Concrete steps:
1. `pip install fastapi uvicorn[standard] websockets` (add to `pyproject.toml [project.optional-dependencies].demo`).
2. Implement `app = FastAPI()` + `@app.websocket("/coach")` handler.
3. On each inbound message `{type:"event", event:{...}}` append to an `ActivityLog`, build a `CoachObservation` via `coach_io.observation_from_log`, call the chosen backend, emit `CoachDecision.to_dict()` if non-`NO_ACTION`.
4. For the LLM backend: hand-craft a tight system prompt (~40 lines) that includes the action space, the Franz-never-advisor rule, Peter-pre-S4 callback rule, and a JSON-only output requirement. Validate the parsed action with `coach.validate_output(action, persona_hint)`.
5. Reuse the existing 5 demo decision JSONs (`sim_loop/replay/src/coachWidgets.tsx*.json`) as **render templates** — the backend picks the right one by `intent` and merges in dynamic copy. This avoids hand-building json-render specs server-side.
6. Smoke test: `pytest tests/test_coach_bridge.py` — one test, hits the WS, fires a synthetic premium-click event, asserts an `upgrade_explain` decision comes back.
7. Run: `uvicorn uniqa.coach_bridge:app --port 8765 --reload`.

- Files touched: `sim_loop/coach.py` (new), `pyproject.toml`, `tests/test_coach_bridge.py` (new, 1 test).
- Acceptance: `wscat -c ws://localhost:8765/coach?persona=franz&backend=rule` returns a decision JSON within 500ms after firing a `premium_click` event.

---

### T3 — Wire React webapp ↔ bridge (01:45–02:45, LOCAL, 60m)

Replace the mock stream in the FunnelTwin mode with the live WS, so the existing
`CoachLayer` renders real decisions. Keep the demo buttons as a fallback.

1. In `sim_loop/replay/src/App.jsx`, `CoachLayerInner`: if `?backend=live` in URL, instantiate `wsStream("ws://localhost:8765/coach?persona=${persona}")` instead of `createLocalMockStream()`. The `wsStream` function already exists in `sim_loop/replay/src/coachWidgets.tsx` (verified). 
2. Wire the recorder to **push every event** to the WS, not just subscribe. Add a tiny `emitToBridge(ev)` callback in the Recorder (`sim_loop/replay/src/capture.js`) that the App.jsx hooks up when `?backend=live`.
3. Add a "Live coach" toggle button next to the existing Demo buttons so the demo operator can switch backends mid-session if the LLM flakes.
4. Manual smoke: open `http://localhost:5173/?mode=twin&backend=live&persona=franz`, click Premium tariff at S4, see the `upgrade_explain` widget appear from the bridge (not from a button press).

- Files touched: `sim_loop/replay/src/App.jsx` (~30 LOC), `sim_loop/replay/src/capture.js` (~10 LOC), one new "Live" badge in the sidebar.
- Acceptance: live mode produces at least 2 distinct decisions across a Franz S1→S6 walk; rule-mode and llm-mode both work; existing demo buttons still fire correctly.

---

### T4 — Live uplift sidebar + persona-autorun (02:45–03:30, DEMO, 45m)

So the judge sees *numbers*, not just one widget pop.

1. **Uplift sidebar:** in App.jsx FunnelTwin mode, add a right-rail card showing the **precomputed** Monte-Carlo A/B numbers (`Overall: 5.6% → 14.5%`, per-persona breakdown). Source: run `python -m calculator.journey -n 4000 --json > sim_loop/replay/public/ab_results.json` once and fetch it. *Don't compute live.*
2. **Persona-autorun mode:** add a "▶ Auto-play as Franz" button that fires a scripted sequence of events (S1 valid → S2 valid → S3 fill → S4 premium_click → wait → S6 hesitate) over ~15 seconds, so a judge sees the coach reacting in real time without manual clicking. Scripts live in `sim_loop/replay/src{judith,franz,peter}.ts`. Three scripts, one per persona, ~20 events each, hard-coded delays.
3. The autoplay is what the operator clicks during the demo: pick persona → autoplay → coach interventions render live → narrate.

- Files touched: `sim_loop/replay/src/App.jsx`, `sim_loop/replay/public/ab_results.json` (new), `sim_loop/replay/src*.ts` (new).
- Acceptance: clicking "Auto-play as Franz" with live backend produces ≥2 coach interventions within 15s, the uplift card shows Coach-off vs Coach-on numbers, and the same flow visibly differs for Peter (callback offer before S4) and Judith (graceful advisor option at S6).

---

### T5 — SLEEP (03:30–08:00, 4.5h)

Set an alarm. If T0–T4 ran long, sleep at least 3h — diminishing returns past 04:30 with a 10:00 deadline. **Do not skip sleep to chase polish.**

---

### T6 — Harvest Leonardo (08:00–08:30, LEO, 30m)

- `ssh leonardo`, `squeue --me`, look for `slurm-persona-<jobid>.out`.
- If job COMPLETED and eval ran: scp the report; pull one number into README ("Persona LoRA on Qwen2.5-1.5B trained on Leonardo A100, eval ε=X").
- If job FAILED / still PENDING / OOM: cancel, delete the bullet from README, move on. **Hard cap 30 min** on this task.

- Acceptance: either a number in the README under "HPC", or the bullet is gone.

---

### T7 — Final A/B + README (08:30–09:15, DEMO, 45m)

1. Re-run `python -m calculator.journey -n 4000 --json > sim_loop/replay/public/ab_results.json` against current code (in case any constant moved).
2. Update `README.md` "Simulation results" table with the current numbers.
3. Add a **"Demo" section** at the top: `streamlit run` removed if not used; the *live demo URL* and steps to reproduce (`uvicorn …`, `npm run dev`, browser URL, autoplay flow).
4. Add a one-paragraph "What the judge sees in 60s" at the top of README.
5. Sanity: `pytest -q` still green (≤ 80 tests).

- Files touched: `README.md`, `sim_loop/replay/public/ab_results.json`.
- Acceptance: README opens with a runnable demo recipe; A/B table reflects current code; tests green.

---

### T8 — Screencast fallback (09:15–09:45, DEMO, 30m)

Record a 60–90s screen capture of the live demo: pick Franz → autoplay → coach intervenes → switch to Peter → callback widget → uplift card. Save as `sim_loop/replay/public/demo.mp4`, link from README. **This is the fallback if the live demo flakes during judging.**

---

### T9 — Submit (09:45–10:00, DEMO, 15m)

Push, tag, fill submission form, send link. Done.

---

## 3. Critical path + fallback

**Critical path to a demo-able artifact:** `T0 → T2 → T3 → T4 → T7`. Everything else is supporting.

- T2 (Python bridge) is the **single point of failure**. If it isn't working by **02:00**, hard-fall back to:
  - Keep the existing **mock stream + 5 demo decision buttons** as the live demo.
  - Add the **autoplay** scripts from T4 anyway, but have them call `streamRef.current.fire(decision)` against the local stream instead of the WS. The autoplay then deterministically fires the right canned decision at the right step.
  - The story becomes "the coach decisions are pre-staged; the bridge to a live LLM coach is on the roadmap." Less impressive, still demo-able, still differentiated per persona.

- **If Leonardo fails entirely** (T1 or T6): cut the HPC bullet from README, lean on
  "scaffold ready, dataset shipped, reservation closes Sun 12:00 — distillation run is
  the immediate next step." The shipped product is the coach, not the persona model;
  judges should not penalise this.

- **If OpenRouter is rate-limited / down**: bridge auto-falls back to `RuleCoachModel`
  (the `?backend=rule` path), which is fully local. Demo continues unchanged; we just
  drop the "LLM coach" line from narration.

---

## 4. Definition of done for submission

### What the demo shows (the 60-second judge story)
1. Open `http://localhost:5173/?mode=twin&backend=live` — the **UNIQA funnel twin** loads, branded.
2. Operator picks persona = **Franz** → clicks **▶ Auto-play**.
3. Franz clicks the **Premium** tariff at S4 → an **`upgrade_explain` coach widget** appears within ~1s, with **human-readable reasoning** visible in the sidebar log. Franz never gets routed to an advisor (hard constraint enforced).
4. Operator switches to **Peter** → autoplay → a **callback / WhatsApp widget** appears *before* the price wall (segment-specific surface choice).
5. Operator switches to **Judith** → autoplay → at the final price reveal, a **`health_explain` + graceful advisor option** widget appears.
6. The **right-rail card** shows the precomputed A/B: **Overall conversion 5.6% → 14.5%, 1.2 avg interventions / 3 budget**.
7. Operator points at `docs/PIPELINE_PLAN.md` and `docs/ARCHITECTURE.md` for the self-improvement story (autoresearch empirical gate + Leonardo distillation path).

### What the README / repo conveys to judges
- ✅ Top-of-README **"What the judge sees in 60s"** + reproducible demo recipe (`uvicorn` + `npm run dev` + URL).
- ✅ **Architecture diagram** (already in `docs/ARCHITECTURE.md`).
- ✅ **A/B uplift table** with the current numbers.
- ✅ **Three differentiated personas** with strategy table (already in README).
- ✅ **Self-improvement story:** autoresearch loop + empirical gate, with Z3 explicitly flagged DEFERRED.
- ✅ **HPC story** (if T6 succeeds): "3 personas distilled to Qwen2.5-1.5B LoRA on CINECA Leonardo, eval ε=X". Otherwise: scaffold present, run pending.
- ✅ **Tests green**, `pytest -q` reproducible.
- ✅ **Screencast** (`sim_loop/replay/public/demo.mp4`) linked from README as judging fallback.
- ✅ **Honest scope:** "Synthetic-data-only validation — no live customer experimentation."

### Hard gates that must hold at submission
- `pytest -q` passes.
- `npm test` in `sim_loop/replay/` passes (parity tests for twin already green per current state).
- Live demo runs end-to-end at least once on the operator's laptop **before** judging starts.
- Screencast exists.
- README's "Quickstart" copy-paste actually works.

---

## Risks (eyes open)

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Coach bridge (T2) takes >2h | Medium | Hard cap 02:00 → fall back to mock-stream + autoplay (still a demo) |
| OpenRouter rate-limit during judging | Low-Med | Bridge has `?backend=rule` fallback baked in |
| Leonardo queue stalls / OOM | Medium | Fire-and-forget; cut bullet at T6 if not done |
| LLM coach emits invalid JSON / wrong action | Med | `coach.validate_output()` gate + retry-once + fall back to rule decision |
| Autoplay timing flaky across browsers | Low | Hard-coded `setTimeout`s; test on the demo laptop before T8 |
| Sleep deprivation tanks T7/T8 polish | Med | Sleep is on the schedule. Honour it. |
| Persona regen ε regresses (already 0.128 at N=500) | Low | Ship N=300 dataset if needed — already proven ε=0.10 |
| Webapp parity tests break after App.jsx edits | Low | Keep edits inside the `?mode=twin&backend=live` branch; legacy capture app untouched |

---

**Stop planning. Start T0 at 23:15.**
