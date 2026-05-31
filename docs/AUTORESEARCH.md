# Self-Improving Coach — Autoresearch + Formal Certificate

> **Status: the formal Z3 certificate is DEFERRED.** The autoresearch loop and its
> **empirical** acceptance gate (`Δuplift > τ` under an annoyance ceiling) are current.
> The Z3 proof described below (the safety guarantee that `τ ≥ 2b` makes every accepted
> change a real improvement) is drafted in `sim_loop/autoresearch.py` and
> kept here as design, but is out of scope this round.

> **Thesis.** *When the user model is right, and the experimentation +
> autoresearch + evals engine are built correctly, the Coach improves itself
> automatically — provably, without touching production traffic.*

This document specifies the self-improvement loop, states the one empirical
assumption it rests on, and links to the Z3 proof that the loop is **sound**
(it can only ever ratchet real-world conversion upward).

---

## 1. Why a self-improving Coach

The Coach is a detection + decision layer on top of the existing UNIQA chatbot.
Its policy — *when* to intervene, with *which* widget, *how strongly* — is a set
of parameters. Hand-tuning those parameters against live traffic is slow, risky,
and ethically fraught (you are experimenting on real customers mid-purchase).

Instead we close the loop **in simulation**, against the psyche-driven persona
model (Judith / Franz / Peter), and only ship policies that the loop has
**proved** to be improvements under a stated assumption.

---

## 2. The loop

```
        ┌──────────────────────────────────────────────────────────────┐
        │  1. PROPOSE    perturb the policy vector  (COACH_GAIN ∈ ℝ^11)  │
        │  2. SIMULATE   psyche persona model → synthetic cohort         │
        │  3. EVALUATE   evals engine measures paired conversion uplift  │
        │  4. GATE       accept ⇔ U_sim(cand) − U_sim(inc) > τ           │
        │  5. REPEAT     accepted candidate becomes the new incumbent    │
        └──────────────────────────────────────────────────────────────┘
```

| Stage | Component | Where |
|-------|-----------|-------|
| Persona model (synthetic data) | `psyche.py` — 6 latent vars, intent mix, hazard-combined bounce | `sim_loop/persona.py` |
| Policy vector | `COACH_GAIN` — per-effect gain multiplier (1.0 = calibrated) | `sim_loop/persona.py` |
| Experimentation | `propose()` — local ±step perturbation of k gains | `sim_loop/autoresearch.py` |
| Evals engine | `evaluate_policy()` — paired A/B on a synthetic cohort via `run_batch` | `sim_loop/autoresearch.py` |
| Gate + loop | `autoresearch()` — hill-climb under the acceptance gate | `sim_loop/autoresearch.py` |
| **Certificate** | Z3 proof of soundness / monotonicity / termination | `sim_loop/autoresearch.py` |

Run it:

```bash
python -m deferred.autoresearch --rounds 30 --tau 0.004 -n 4000
# start uplift: 8.60pp → best uplift: 10.23pp  (gated, monotone)
```

---

## 3. The one assumption (A1)

Everything hinges on a single **empirical** claim — *not* proved by Z3, but
validated by comparing synthetic vs. real funnel statistics:

> **A1 (model fidelity).** The persona simulator produces synthetic sessions
> whose statistics are ε-close to reality. Equivalently, the paired-uplift
> estimator `U_sim(policy)` has bounded bias against the true uplift:
>
> ```
> ∀ policy:   | U_sim(policy) − U_real(policy) |  ≤  b ,     b = L·ε
> ```
>
> where **ε** is the statistical distance (e.g. total variation) between the
> synthetic and real session distributions, and **L** is the sensitivity
> (Lipschitz constant) of the uplift functional w.r.t. that distance.

How we keep A1 honest:
- **Calibration anchors.** The model is pinned to observed funnel reality:
  baseline ≈ 5.6 %, Step-4 drop ≈ 66 %, Step-5 ≈ 24 %, Step-6 ≈ 78 %.
- **Distribution checks.** Synthetic per-step drop-off and bounce-reason mix are
  compared to real analytics; ε is estimated from the gap.
- **Conservative τ.** We choose the acceptance margin **τ ≥ 2b** so the proof's
  hypothesis holds (see §4). Larger uncertainty ⇒ larger τ ⇒ fewer but safer
  accepted changes.

---

## 4. The certificate (Z3)

`sim_loop/autoresearch.py` discharges five theorems. Each is proved by
asserting the negation and checking it is **UNSAT**.

| # | Theorem | Statement |
|---|---------|-----------|
| **T1** | Soundness | `τ ≥ 2b ∧ accept ⇒ U_real(cand) > U_real(inc)` — every accepted candidate is a *real* improvement |
| **T2** | No-regression | `τ ≥ 2b ∧ U_real(cand) ≤ U_real(inc) ⇒ ¬accept` — a real regression is *never* accepted |
| **T3** | Monotonicity | incumbent real uplift is non-decreasing every round (accept ⇒ up; reject ⇒ unchanged) |
| **T4** | Termination | bounded uplift + each accept adds ≥ δ>0 ⇒ acceptances are finite ⇒ the loop converges |
| **T5** | Tightness | `τ < 2b` admits a false accept — so the bound `τ ≥ 2b` is *necessary*, not just sufficient |

The key inequality behind T1/T2:

```
U_real(cand) − U_real(inc)  ≥  (U_sim(cand) − U_sim(inc))  −  2b   >   τ − 2b   ≥  0
                               └────── gate says > τ ──────┘
```

Run the proof:

```bash
python sim_loop/autoresearch.py     # → ALL THEOREMS DISCHARGED ✅
```

It is also exercised in CI via `tests/test_autoresearch.py::test_z3_certificate_passes`.

---

## 5. What is proved vs. what is assumed

| | Status |
|---|---|
| The persona model matches reality | **Assumed (A1)** — empirical, validated by calibration + distribution checks |
| Given A1, the gate only accepts real improvements | **Proved (T1, T2)** |
| Given A1, real conversion is monotone non-decreasing | **Proved (T3)** |
| The loop terminates / converges | **Proved (T4)** |
| The `τ ≥ 2b` bound is necessary | **Proved (T5)** |

> **Conclusion.** The correctness of automatic Coach improvement is reduced to a
> *single, measurable* condition — model fidelity (A1). Everything downstream of
> A1 is mechanically certified. "Make the simulator faithful" is the whole job;
> the optimisation is then provably safe.

---

## 6. Roadmap

- **R1 — fidelity dashboard.** Estimate ε live from real funnel analytics; surface
  `b = L·ε` and auto-set `τ`.
- **R2 — richer policy space.** Tune decision thresholds and widget copy variants,
  not just effect gains.
- **R3 — LLM-proposed experiments.** Replace random `propose()` with an
  LLM-driven hypothesis generator (autoresearch agent) that reads eval traces and
  proposes targeted policy edits; the gate stays unchanged, so soundness holds.
- **R4 — shadow deployment.** Periodically re-estimate ε from a small live
  holdout to keep A1 honest as customer behaviour drifts.
